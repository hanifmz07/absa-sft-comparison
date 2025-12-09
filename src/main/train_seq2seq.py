import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate
import numpy as np

# 1. Load Dataset (English-Indonesian translation example)
# We use a small subset for demonstration
dataset = load_dataset("opus_books", "en-id")
dataset = dataset["train"].train_test_split(test_size=0.2)

# 2. Load Tokenizer & Model
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 3. Preprocessing
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "id"

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True
    )

    # Tokenize targets (labels)
    # We use text_target=... to tokenize the target text
    labels = tokenizer(
        text_target=targets, 
        max_length=max_target_length, 
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. Metric Computation (BLEU score)
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
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

# 5. Training Arguments
args = Seq2SeqTrainingArguments(
    output_dir="./mt5-finetuned-en-id",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduce if OOM
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,     # Essential for Seq2Seq metrics
    fp16=torch.cuda.is_available(), # Use mixed precision if on GPU
)

# 6. Data Collator
# This handles dynamic padding (crucial for efficiency)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 7. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train
trainer.train()

# 9. Inference Example
input_text = "Hello, how are you today?"
inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
print("Translated:", tokenizer.decode(outputs[0], skip_special_tokens=True))