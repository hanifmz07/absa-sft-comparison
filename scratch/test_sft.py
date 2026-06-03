import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

data = [{"input": "the room is good and suitable for the budget . [A] [O] [S]", "target": "[A] room [O] good [S] positive [SSEP] [A] room [O] suitable for the budget [S] positive"}]
df = pd.DataFrame(data)

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

print(trainer.train_dataset[0])
