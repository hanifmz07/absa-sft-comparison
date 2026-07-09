from ..utils.eval_utils import calculate_metrics
import torch
import json
import time
from tqdm import tqdm
import os, re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils.constrained_decoding import MVPConstrainedDecoder, GASConstrainedDecoder, LegoABSAConstrainedDecoder
from ..utils.io_utils import load_json_with_fallback


def resolve_output_dir(base_output_dir, model_path, use_constrained_decoding):
    base_output_dir = base_output_dir.rstrip("/")
    model_name = os.path.basename(model_path.rstrip("/"))

    # Avoid duplicating checkpoint/model folder when caller already includes it in output_dir.
    if os.path.basename(base_output_dir) == model_name:
        eval_dir = base_output_dir
    else:
        eval_dir = os.path.join(base_output_dir, model_name)

    return os.path.join(
        eval_dir,
        "constrained_decoding" if use_constrained_decoding else "unconstrained_decoding",
    )

def main(args):
    # === Load Model ===
    print(f'Model Path: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    torch_dtype = getattr(torch, args.torch_dtype) if args.torch_dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation
    )

    print("Tokenizer length:", len(tokenizer))
    print("Embedding size:", model.get_input_embeddings().weight.shape)
    print("Special Tokens:", tokenizer.special_tokens_map)

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # === Setup Device ===
    model.eval()

    # === Load Dataset ===
    test_data = load_json_with_fallback(args.test_json_path)
    if args.limit_samples is not None:
        test_data = test_data[:args.limit_samples]

    # Get the input prompts, labels, sentence IDs, and task elements
    prompts = [f'{instance["input"]} =>' for instance in test_data]
    labels = [instance['target'] for instance in test_data]
    sentence_ids = [instance['sentence_id'] for instance in test_data]
    tasks = [instance['task_elements'] for instance in test_data]
    element_orders = [instance['element_order'] for instance in test_data]

    # --- Batching Modification ---
    batch_size = args.batch_size # You can adjust this based on your GPU memory
    outputs = []

    start_time = time.perf_counter()
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)

        logits_processor = None
        if args.use_constrained_decoding:
            # Setup constrained decoding based on prompt type
            if args.prompt_type == "mvp":
                logits_processor = MVPConstrainedDecoder(inputs['input_ids'], tokenizer)
            elif args.prompt_type == "gas":
                logits_processor = GASConstrainedDecoder(inputs['input_ids'], tokenizer)
            elif args.prompt_type == "legoabsa":
                logits_processor = LegoABSAConstrainedDecoder(inputs['input_ids'], tokenizer)
            else:
                raise ValueError(f"Unknown prompt type for constrained decoding: {args.prompt_type}")
            
        # Generate outputs for the batch
        batch_outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            # stop_at_eos=True,
            do_sample=False,
            logits_processor=[logits_processor] if logits_processor else None,
            pad_token_id=tokenizer.pad_token_id,
            # return_type="str",
            # verbose=False,
        )
        batch_outputs.to(device="cpu")
        # print(batch_outputs)
        batch_outputs_text = tokenizer.batch_decode(
            batch_outputs[:, inputs['input_ids'].shape[1]:],  # Slice to remove prompt
            skip_special_tokens=False if args.prompt_type == "legoabsa" else True,
        )
        # Remove '<|endoftext|>' tokens if present
        batch_outputs_text = [output.replace(tokenizer.eos_token, '').strip() for output in batch_outputs_text]
        
        if args.debug_generations:
            for prompt, output in zip(batch_prompts, batch_outputs_text):
                print(f"Prompt: {prompt}")
                print(f"Generated: {output}")
                print("-" * 50)

        # if isinstance(batch_outputs_text, str):
            # batch_outputs_text = [batch_outputs_text]

        outputs.extend(batch_outputs_text)

    end_time = time.perf_counter()
    total_duration = end_time - start_time
    # # Cut off the prompts from the outputs
    # outputs = [output[len(prompts[idx]):].strip() for idx, output in enumerate(outputs)]

    # Postprocess outputs and calculate metrics
    # Temporary storing for debugging
    output_dir = resolve_output_dir(args.output_dir, args.model_path, args.use_constrained_decoding)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "raw_inference_results.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
 
    inference_results = []

    per_task = {}
    for prompt, pred, label, si, t, element_order in zip(prompts, outputs, labels, sentence_ids, tasks, element_orders):
        
        if args.prompt_type == "mvp":
            # Postprocess the prediction for MvP
            # Split the target and prediction into lists
            target_split = label.split("[SSEP]")
            pred_split = pred.split("[SSEP]")
            # Strip whitespace
            target_split = [l.strip() for l in target_split]
            pred_split = [l.strip() for l in pred_split]

            # Make sure there is a whitespace before and after [A], [O], and [S]
            target_split = [l.replace("[A]", " [A] ").replace("[O]", " [O] ").replace("[S]", " [S] ").strip() for l in target_split]
            pred_split = [l.replace("[A]", " [A] ").replace("[O]", " [O] ").replace("[S]", " [S] ").strip() for l in pred_split]

            # Replace double or more whitespaces with a single whitespace
            target_split = [re.sub(r'\s+', ' ', l) for l in target_split]
            pred_split = [re.sub(r'\s+', ' ', l) for l in pred_split]

        elif args.prompt_type == "gas":
            # Split the target and prediction into lists (GAS and LegoABSA)
            target_split = label.split(';')
            pred_split = pred.split(';')
            target_split = [l.strip() for l in target_split]
            pred_split = [l.strip() for l in pred_split]

            # Make sure there is a whitespace before and after '(' and '|'
            target_split = [l.replace('(', ' ( ').replace(')', ' ) ').replace('|', ' | ').strip() for l in target_split]
            pred_split = [l.replace('(', ' ( ').replace(')', ' ) ').replace('|', ' | ').strip() for l in pred_split]

            # Replace double or more whitespaces with a single whitespace
            target_split = [re.sub(r'\s+', ' ', l) for l in target_split]
            pred_split = [re.sub(r'\s+', ' ', l) for l in pred_split]

        elif args.prompt_type == "legoabsa":
            # Split the target and prediction into lists (GAS and LegoABSA)
            target_split = label.split(';')
            pred_split = pred.split(';')
            target_split = [l.strip() for l in target_split]
            pred_split = [l.strip() for l in pred_split]

            # Make sure there is a whitespace before and after '(' and '|'
            target_split = [l.replace('<|aspect|>', ' <|aspect|> ').replace('<|opinion|>', ' <|opinion|> ').replace('<|sentiment|>', ' <|sentiment|> ').strip() for l in target_split]
            pred_split = [l.replace('<|aspect|>', ' <|aspect|> ').replace('<|opinion|>', ' <|opinion|> ').replace('<|sentiment|>', ' <|sentiment|> ').strip() for l in pred_split]

            # Replace double or more whitespaces with a single whitespace
            target_split = [re.sub(r'\s+', ' ', l) for l in target_split]
            pred_split = [re.sub(r'\s+', ' ', l) for l in pred_split]
            
        else:
            raise ValueError(f"Unknown prompt type: {args.prompt_type}")

        # Store inference results
        inf_dict = {}
        inf_dict["sentence_id"] = si
        inf_dict["task_elements"] = t
        inf_dict["element_order"] = element_order
        inf_dict["input"] = prompt
        inf_dict["target"] = label
        inf_dict["prediction"] = pred
        inf_dict["target_list"] = target_split
        inf_dict["prediction_list"] = pred_split
        inf_dict["inference_time"] = total_duration

        inference_results.append(inf_dict)

        # Store predictions and targets per task for metric calculation
        if element_order not in per_task.keys():
            per_task[element_order] = {"predictions": [], "targets":[]}
        per_task[element_order]["predictions"].append(pred_split)
        per_task[element_order]["targets"].append(target_split)


    result_metrics = {}
    for task, v in per_task.items():
        predictions = v["predictions"]
        targets = v["targets"]
        result_metrics.update(
            calculate_metrics(predictions, targets, task)
    )
    scaled_result_metrics = {key: value * 100 for key, value in result_metrics.items()}
    scaled_result_metrics["inference_duration"] = total_duration
    
    print("Evaluation results:", scaled_result_metrics)

    # Save the metric results
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(scaled_result_metrics, f, indent=4, ensure_ascii=False)
    print(f"Evaluation results saved to {output_file}")

    # Save inference results if specified
    if args.save_predictions:
        inference_output_file = os.path.join(output_dir, "inference_results.json")
        with open(inference_output_file, "w", encoding='utf-8') as f:
            json.dump(inference_results, f, indent=4, ensure_ascii=False)
        print(f"Inference results saved to {inference_output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate models for ABSA task")
    parser.add_argument("--test_json_path", type=str, required=True, help="Path to the test JSON dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--prompt_type", type=str, required=True, help="Prompt type for the model, either mvp, gas, or legoabsa.", choices=["mvp", "gas", "legoabsa"])
    parser.add_argument("--output_dir", type=str, default=f"./outputs/evals", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--save_predictions", action="store_true", help="Save inference results to a JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum number of new tokens to generate")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit the number of samples to evaluate")
    parser.add_argument("--debug_generations", action="store_true", help="Print generations for debugging")
    parser.add_argument("--torch_dtype", type=str, default="auto", help="Dtype to load the model in (e.g. auto, float32, bfloat16, float16). 'auto' matches the checkpoint's saved dtype.")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="Attention implementation to use (e.g. eager, sdpa).")

    # Constrained decoding arguments
    parser.add_argument("--use_constrained_decoding", action="store_true", help="Whether to use constrained decoding during generation")

    args = parser.parse_args()
    main(args)
