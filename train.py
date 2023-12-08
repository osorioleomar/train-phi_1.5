! pip install accelerate transformers einops datasets peft bitsandbytes

from huggingface_hub import notebook_login
notebook_login()

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5",
    device_map={"":0},
    trust_remote_code=True,
    quantization_config=bnb_config
)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["Wqkv", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

def tokenize(sample):
    model_inps =  tokenizer(sample["text"], padding=True, truncation=True, max_length=512)
    return model_inps

data = load_dataset("gsm8k", "main", split="train")
data_df = data.to_pandas()
data_df["text"] = data_df[["question", "answer"]].apply(lambda x: "question: " + x["question"] + " answer: " + x["answer"], axis=1)
data = Dataset.from_pandas(data_df)
tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)
tokenized_data

training_arguments = TrainingArguments(
        output_dir="phi-1_5-finetuned-gsm8k",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=100,
        max_steps=1000,
        num_train_epochs=1,
        push_to_hub=True
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
trainer.push_to_hub()

