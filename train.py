# Configurable Parameters
MODEL_ID = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "eemotgs/en_es_orca_tiny"

import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling
from torch import cuda, bfloat16
import transformers

# Get Model
from huggingface_hub import notebook_login


model_id = MODEL_ID

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=True
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=True
)
model.eval()
print(f"Model loaded on {device}")
mem = model.get_memory_footprint()
print("Memory footprint: {} ".format(mem))

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True
)

# Get Dataset
from datasets import load_dataset

dataset = load_dataset(DATASET_NAME, split="train")
print(f'Number of records: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

# dataset_cot = dataset.filter(lambda example: example['type'] == "CoT")
dataset_cot = dataset
print(f'Number of records: {len(dataset_cot)}')
print(f'Column names are: {dataset_cot.column_names}')


def create_prompt(rec):
    start = "Read the Instruction below and provide an answer."
    question = f"### INSTRUCTION:\n{rec['source']}\n\n"
    response = f"### RESPONSE:\n{rec['rationale']}\n"
    answer = f"Therefore the answer is {rec['target']}\n\n"
    end = "### End"

    parts = [part for part in [start, question, response, answer, end] if part]

    formatted_prompt = "\n\n".join(parts)
    formatted_prompt = formatted_prompt.replace('\\n', '\n')

    rec["text"] = formatted_prompt

    return rec


p = create_prompt(dataset_cot[4999])
print(p)
print(p["text"])
dataset = dataset_cot.map(create_prompt)
dataset = dataset.map(
    batched=True,
    remove_columns=['source', 'target', 'rationale', 'task', 'type']
)
print(dataset[4999]["text"])


# Save dataset to the hub for future use
# dataset.push_to_hub("Venkat-Ram-Rao/processed_cot_dataset", private=True)

# max length of the model
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


mx = get_max_length(model)
mx
len(dataset)
# tokenize dataset
dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
len(dataset)
dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < mx)
len(dataset)
seed = 42
set_seed(seed)
dataset = dataset.shuffle(seed=seed)

# Freeze Original Weights
for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


model.lm_head = CastOutputToFloat(model.lm_head)


# Create Lora Config
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


modules = find_all_linear_names(model)
print(modules)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,  # attention heads
    lora_alpha=64,  # alpha scaling
    target_modules=modules,  # gonna train all
    lora_dropout=0.1,  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",  # for Decoder models like GPT Seq2Seq for Encoder-Decoder models like T5
)

## Get the PEFT Model using the downloaded model and the loRA config
model = get_peft_model(model, config)

# Training
# Print Trainable parameters
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
)

tokenizer.pad_token = tokenizer.eos_token
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

trainer.train()
