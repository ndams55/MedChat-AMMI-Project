from huggingface_hub import login
login()

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from transformers import TrainingArguments
from transformers import Trainer
import pandas as pd

from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

import math





df_train = pd.read_csv('/Users/ndjebayidamarisstephanie/Projects/LLM-Project/French-Med-Bot/data/french-train.csv')
df_test = pd.read_csv('/Users/ndjebayidamarisstephanie/Projects/LLM-Project/French-Med-Bot/data/french-test.csv')

datasets = load_dataset('csv', data_files={"train": '/Users/ndjebayidamarisstephanie/Projects/LLM-Project/French-Med-Bot/data/french-train.csv', "validation": '/Users/ndjebayidamarisstephanie/Projects/LLM-Project/French-Med-Bot/data/french-test.csv'})


model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def preprocess_function(examples):
    questions = [q.strip() for q in examples["short_question"]]
    inputs = tokenizer(
        questions,
        examples["short_answer"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    return inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

from transformers import TrainingArguments

output_dir = "/kaggle/working/"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
evaluation_strategy="epoch"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
weight_decay=0.01
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"
num_train_epochs = 5
#clf_metrics = evaluate.combine(["f1", "precision", "recall"])

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy=evaluation_strategy,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    num_train_epochs=num_train_epochs,
    lr_scheduler_type=lr_scheduler_type,
    push_to_hub=True,
    logging_strategy="epoch", #Extra: to log training data stats for loss
    #compute_metrics=compute_metrics,
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["short_question"])):
        text = f"### Question: {example['short_question'][i]}\n ### Answer: {example['short_answer'][i]}"
        output_texts.append(text)
    
    return output_texts


max_seq_length = 512
#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=datasets['train'],
    eval_dataset=datasets['validation'],
    formatting_func=formatting_prompts_func,
    #data_collator=data_collator,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    peft_config=peft_config,
)

trainer.train() 



eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")