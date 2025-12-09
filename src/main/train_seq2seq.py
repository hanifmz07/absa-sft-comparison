import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datetime import datetime
import evaluate
import numpy as np
import argparse
import wandb
import os
import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv()

def main(args):
    def compute_metrics(eval_preds):
        metric = evaluate.load("sacrebleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode generated predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Post-process for SacreBLEU (requires list of list for references)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}
    
    def print_config(args):
        print("=" * 20 + " Training Configuration " + "=" * 20)
        print(f"Model Name: {args.model_name}")
        print(f"Training Dataset Path: {args.train_json_path}")
        if args.val_json_path is not None:
            print(f"Validation Dataset Path: {args.val_json_path}")
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
        print(f"Output Directory: {output_dir}")
        print(f"Output Folder Name: {output_folder_name}")
        print("=" * 60)

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

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=os.getenv("HF_CACHE_DIR"))
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, dtype=torch.bfloat16 if bf16 else torch.float16, trust_remote_code=True, device_map="auto", cache_dir=os.getenv("HF_CACHE_DIR"))

    def preprocess_function(instance):
        # Tokenize inputs
        model_inputs = tokenizer(
            instance['input'] + " =>",  # Includes the separator
            # truncation=True
        )

        # Tokenize targets (labels)
        # We use text_target=... to tokenize the target text
        labels = tokenizer(
            text_target= " " + instance['target'], 
            # truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Load dataset
    print(f"Loading dataset from {args.train_json_path}...")
    with open(args.train_json_path, 'r') as f:
        json_data = json.load(f)

    df = pd.DataFrame(json_data)
    # Sample dataset if specified
    if args.sample_size is not None:
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(preprocess_function)

    # Load validation dataset if provided
    if args.val_json_path is not None and args.eval_strategy != "no":
        print(f"Loading validation dataset from {args.val_json_path}...")
        with open(args.val_json_path, 'r') as f:
            val_json_data = json.load(f)
        
        val_df = pd.DataFrame(val_json_data)
        # val_df["text"] = val_df.apply(lambda row: f"{row['input']} => {row['target']}", axis=1)
        val_dataset = Dataset.from_pandas(val_df)
        val_dataset = val_dataset.map(preprocess_function)

    # Training Arguments
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

    print_config(args)

    # Set the project name where your runs will appear in the dashboard
    os.environ["WANDB_PROJECT"] = "absa-seq2seq"

    model_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=args.batch_size,  # Reduce if OOM
        learning_rate=args.lr,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,     # Essential for Seq2Seq metrics
        bf16=bf16, # Use mixed precision if on GPU
        optim=args.optimizer,
        # completion_only_loss=True,

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

    # Data Collator
    # This handles dynamic padding (crucial for efficiency)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=model_args,
        train_dataset=dataset,
        eval_dataset=val_dataset if args.val_json_path is not None and args.eval_strategy != "no" else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("--- Starting Seq2SeqTrainer ---")
    history = trainer.train()
    print(history)
    print(f"--- Training complete at {datetime.now().strftime('%Y%m%d_%H%M%S')}---")
    print(f"Duration: {datetime.now() - datetime.strptime(current_time, '%Y%m%d_%H%M%S')}")
    print(f"Model saved to {output_dir}")

    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json_path", type=str, required=True, help="Path to the training JSON dataset")
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./outputs/models", help="Output directory")
    parser.add_argument("--prompt_type", type=str, choices=["gas", "mvp", "mvp_aos", 'legoabsa'], default="mvp_aos", help="Sampling/prompt style")
    
    parser.add_argument("--save_strategy", type=str, choices=["epoch", "best", "no"], default='best', help="Model saving mode during training")

    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer to use (See `OptimizerNames` (https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    parser.add_argument("--eval_strategy", type=str, choices=["no", "steps", 'epoch'], default="epoch", help="Validation mode during training")
    parser.add_argument("--val_json_path", type=str, required=False, help="Path to the validation JSON dataset")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Batch size for validation")
    
    parser.add_argument("--sample_size", type=int, default=None, help="Target number of training instances after sampling (None = use all)")

    args = parser.parse_args()

    main(args)
