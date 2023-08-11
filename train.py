# Configurable Parameters
# MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_ID = "projecte-aina/aguila-7b"
DATASET_NAME = "eemotgs/en_es_orca_tiny"
output_dir = "outputs"

import bitsandbytes as bnb
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling
from torch import cuda, bfloat16
import transformers
import os
from peft import PeftModel

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback

from datasets import load_dataset
import random

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

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True
)

# Get Dataset

dataset = load_dataset(DATASET_NAME, split="train")
print(f'Number of records: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

tmp_dataset = dataset


def create_prompt(rec):
    start = f"### SYSTEM PROMPT:\n{rec['system_prompt']}\n\n"
    question = f"### INSTRUCTION:\n{rec['question']}\n\n"
    response = f"### RESPONSE:\n{rec['response']}\n"
    end = "### End"

    parts = [part for part in [start, question, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    formatted_prompt = formatted_prompt.replace('\\n', '\n')

    rec["text"] = formatted_prompt

    return rec


p = create_prompt(tmp_dataset[99])
dataset = tmp_dataset.map(create_prompt)
dataset = dataset.map(
    batched=True,
    remove_columns=['system_prompt',
                    'question', 'response', 'id']

)


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



# tokenize dataset
dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < mx)

seed = random.randint(1, 99)
set_seed(seed)
dataset = dataset.shuffle(seed=seed)

# Freeze Original Weights
for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
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

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


modules = find_all_linear_names(model)
print(modules)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=modules,  # should we  train all?
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)


## Saving after training

def save_model(args, state, kwargs):
    print('Saving PEFT checkpoint...')
    if state.best_model_checkpoint is not None:
        checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
    else:
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

    peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
    kwargs["model"].save_pretrained(peft_model_path)

    pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    if os.path.exists(pytorch_model_path):
        os.remove(pytorch_model_path)


class SavePeftModelCallback(transformers.TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        save_model(args, state, kwargs)


# Training

tokenizer.pad_token = tokenizer.eos_token
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=10,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        learning_rate=1e-4,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[SavePeftModelCallback]
)

model.config.use_cache = False

# trainer.train()

model.save_pretrained(output_dir)

device = "auto"  # or any specific device name
push_to_hub = True  # or False

device_arg = { 'device_map': 'auto' }


print(f"Loading PEFT: {output_dir}")
model = PeftModel.from_pretrained(model, peft_model_path, **device_arg)
print(f"Running merge_and_unload")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)


model.save_pretrained(f"{output_dir}")
tokenizer.save_pretrained(f"{output_dir}")
print(f"Model saved to {output_dir}")



