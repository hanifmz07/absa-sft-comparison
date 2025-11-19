# You need: pip install transformers datasets trl peft
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
import wandb
import torch

import pickle
import os
import argparse
import json
import pandas as pd

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

def main(args):
    # Set current time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Check capability of GPU, if possible, use bfloat16 (bf16) for faster training
    bf16 = False
    if torch.cuda.is_available():
        gpu_capability = torch.cuda.get_device_capability()
        if gpu_capability[0] >= 8:
            print("GPU supports bf16, using bf16 for training.")
            bf16 = True
        else:
            print("GPU does not support bf16, using fp16 for training.")
    else:
        print("No GPU detected, training will be on CPU.")

    # Load model
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16 if bf16 else torch.float16, trust_remote_code=True, device_map="auto", cache_dir=os.getenv("HF_CACHE_DIR"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=os.getenv("HF_CACHE_DIR"))

    # Add special tokens for LEGO-ABSA if needed
    if 'legoabsa' in args.train_json_path:
        print("Adding LEGO-ABSA special tokens to tokenizer...")
        custom_tokens = ["<|aspect|>", "<|opinion|>", "<|sentiment|>"]

        # Check if the custom tokens is already added to the tokenizer or not
        if all(token in tokenizer.special_tokens_map['additional_special_tokens'] for token in custom_tokens):
            print("Custom tokens already exist in the tokenizer. Skipping addition.")
        else:
            num_added_tokens = tokenizer.add_special_tokens({
                "additional_special_tokens": tokenizer.special_tokens_map['additional_special_tokens'] + custom_tokens
            })

            print(f"Added {num_added_tokens} new tokens.")
            print(f"New tokenizer size: {len(tokenizer)}")

            print(f"Resizing model embeddings to accommodate new tokens...")
            model.resize_token_embeddings(len(tokenizer))
            print(f"New model embedding size: {model.get_input_embeddings().weight.size(0)}")

    # IMPORTANT: Dataset must have 'input' and 'target' fields
    def split_prompt_and_completion(instance):
        return {
            "prompt": instance['input'] + " =>",  # Includes the separator
            "completion": " " + instance['target']            # The part you want to train on
        }
    
    # Load dataset
    print(f"Loading dataset from {args.train_json_path}...")
    with open(args.train_json_path, 'r') as f:
        json_data = json.load(f)

    # Format dataset
    df = pd.DataFrame(json_data)
    # df["text"] = df.apply(lambda row: f"{row['input']} => {row['target']}", axis=1)

    # Sample dataset if specified
    if args.sample_size is not None:
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(split_prompt_and_completion)
    
    # Load validation dataset if provided
    if args.val_json_path is not None and args.eval_strategy != "no":
        print(f"Loading validation dataset from {args.val_json_path}...")
        with open(args.val_json_path, 'r') as f:
            val_json_data = json.load(f)
        
        val_df = pd.DataFrame(val_json_data)
        # val_df["text"] = val_df.apply(lambda row: f"{row['input']} => {row['target']}", axis=1)
        val_dataset = Dataset.from_pandas(val_df)
        val_dataset = val_dataset.map(split_prompt_and_completion)


    # Create output directory with detailed naming
    output_folder_name = f"{current_time}"
    output_folder_name += f"_{os.path.splitext(os.path.basename(args.train_json_path))[0]}"
    output_folder_name += f"_model-{args.model_name.split('/')[-1]}"
    output_folder_name += f"_lr-{args.lr}"
    output_folder_name += f"_bs-{args.batch_size}"
    output_folder_name += f"_epochs-{args.num_epochs}"
    if args.sample_size is not None:
        output_folder_name += f"_samples-{args.sample_size}"
    output_dir = os.path.join(args.output_dir, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Print all configurations
    print("=" * 20 + " Training Configuration " + "=" * 20)
    print(f"Model Name: {args.model_name}")
    print(f"Training Dataset Path: {args.train_json_path}")
    if args.val_json_path is not None:
        print(f"Validation Dataset Path: {args.val_json_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Output Folder Name: {output_folder_name}")
    print(f"Prompt Type: {args.prompt_type}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.num_epochs}")
    if args.sample_size is not None:
        print(f"Sample Size: {args.sample_size}")
    print(f"Save Strategy: {args.save_strategy}")
    print(f"Evaluation Strategy: {args.eval_strategy}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Set the project name where your runs will appear in the dashboard
    os.environ["WANDB_PROJECT"] = "absa-sft"

    # Prepare training arguments
    training_args = SFTConfig(
        # Training settings
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=1e-2,
        num_train_epochs=args.num_epochs,
        bf16=bf16,
        # dataset_text_field="text",
        optim=args.optimizer,
        completion_only_loss=True,

        # Evaluation settings
        eval_strategy=args.eval_strategy,
        per_device_eval_batch_size=args.val_batch_size,
        
        # Logging and saving settings
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",
        save_strategy=args.save_strategy,
        load_best_model_at_end=True if args.save_strategy == "best" else False,
        output_dir=output_dir,
        run_name=f"seed-{args.seed}_optimizer-{args.optimizer}_lr-{args.lr}_samplesize-{args.sample_size if args.sample_size is not None else 'all'}_data-{args.train_json_path.split('/')[3]}_{current_time}",

        # Seed settings
        seed=args.seed,
        data_seed=args.seed,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset if args.val_json_path is not None and args.eval_strategy != "no" else None,
        args=training_args,
    )

    print("--- Starting SFTTrainer ---")
    history = trainer.train()
    print(f"--- Training complete at {datetime.now().strftime('%Y%m%d_%H%M%S')}---")
    print(f"Duration: {datetime.now() - datetime.strptime(current_time, '%Y%m%d_%H%M%S')}")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json_path", type=str, required=True, help="Path to the training JSON dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./outputs/models", help="Output directory")
    parser.add_argument("--prompt_type", type=str, choices=["gas", "mvp", "mvp_aos", 'legoabsa'], default="mvp", help="Sampling/prompt style")
    
    parser.add_argument("--save_strategy", type=str, choices=["epoch", "best", "no"], default='best', help="Model saving mode during training")

    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer to use (See `OptimizerNames` (https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    parser.add_argument("--eval_strategy", type=str, choices=["no", "steps", 'epoch'], default="epoch", help="Validation mode during training")
    parser.add_argument("--val_json_path", type=str, required=False, help="Path to the validation JSON dataset")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Batch size for validation")
    
    parser.add_argument("--sample_size", type=int, default=None, help="Target number of training instances after sampling (None = use all)")
    

    args = parser.parse_args()

    main(args)