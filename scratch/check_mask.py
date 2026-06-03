import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

import json
with open("dataset/hotel_reviews/eng/mvp_aos/train.json") as f:
    json_data = json.load(f)

df = pd.DataFrame(json_data)
def split_prompt_and_completion(instance):
    return {
        "prompt": instance['input'] + " =>",
        "completion": " " + instance['target']
    }

dataset = Dataset.from_pandas(df)
dataset = dataset.map(split_prompt_and_completion)

training_args = SFTConfig(
    output_dir="./tmp",
    completion_only_loss=True,
    use_cpu=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer
)

# Check the formatted dataset directly

# SFTTrainer formats the dataset. Let's check the formatted dataset.
formatted_dataset = trainer.train_dataset

failed_count = 0
for i in range(len(formatted_dataset)):
    mask = formatted_dataset[i]['completion_mask']
    # If the mask has NO 1s (all 0s), it failed to find the completion
    if sum(mask) == 0:
        failed_count += 1
        print(f"Failed at {i}: {dataset[i]['prompt']}")

print(f"Total failed: {failed_count} out of {len(formatted_dataset)}")
