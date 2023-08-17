import torch
import time
import numpy as np
import random
import bitsandbytes as bnb

from tqdm import tqdm
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
    set_seed
)
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pynvml import *

from datasets import set_caching_enabled

set_caching_enabled(False)
set_seed(42)

torch.cuda.empty_cache()

visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")

model_name = "tiiuae/falcon-7b"
datset_name = "Open-Orca/OpenOrca"
optim = "paged_adamw_32bit"
sep = " "

train_on_source = False
learning_rate = 2e-4
epochs = 5
batch_size = 32
lora_r = 16
output_dir = "falcon_lora_ft"
output_fname = f"eval_prediction_orca_bs{batch_size}_r{lora_r}_no_eos__test"


def print_gpu_utilization(device_idx):
    """
    Prints gpu utilization by device number.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(device_idx))
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"device {device_idx}:")
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")
    print(f"GPU memory total: {info.total // 1024 ** 2} MB.")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


for device in visible_devices:
    print_gpu_utilization(device)

# TODO: change to your settings
train_dataset = load_dataset(datset_name, split="train[:1000]")
eval_dataset = load_dataset(datset_name, split="train[1000:1050]")


def evaluate_model(trainer, tokenizer, dataset, output_fname):
    """
    Writes the predicted output into output_fname
    """
    start = time.time()
    trainer.model.config.use_cache = True
    start = time.time()
    predictions = trainer.predict(dataset)
    predictions = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)
    pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    with open(output_fname, "a") as f:
        for i, (input_text, pred_text) in enumerate(zip(dataset["text"], pred_text)):
            f.write(f"EXAMPLE {i}:\n")
            f.write(input_text + pred_text)
            f.write("\n\n")

    print(f"evaliation time per {len(dataset)} samples: {time.time() - start}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

for device in visible_devices:
    print_gpu_utilization(device)


class DataCollatorCustom:

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            max_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]

        max_label_length = max(len(l) for l in labels)
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [-100] * (max_label_length - len(feature["labels"]))
            feature["labels"] = feature["labels"] + remainder if padding_side == "right" else remainder + feature[
                "labels"]

        features = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return features


def get_linear_layers(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto",
                                             trust_remote_code=True)
model.config.use_cache = False

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=get_linear_layers(model),
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

collator = DataCollatorCustom(tokenizer, pad_to_multiple_of=8)


# TODO: specific to the dataset, change or remove
def filtering_function(chunk):
    return [p and q and r for p, q, r in zip(chunk['system_prompt'], chunk['question'], chunk['response'])]


# TODO: specific to the dataset, change according to your format
def preprocess_function(chunk, tokenizer, sep=" ", train_on_source=False):
    input_text = [
        f"### SYSTEM PROMPT: {p}{sep}### INSTRUCTION: {q}{sep} ### RESPONSE: ".replace('\\n', '\n')
        for p, q in zip(chunk['system_prompt'], chunk['question'])
    ]

    input = tokenizer(input_text, truncation=True, max_length=tokenizer.model_max_length)
    target_text = [
        f"{r}{tokenizer.eos_token}"
        for r in chunk['response']
    ]
    target = tokenizer(target_text, truncation=True, max_length=tokenizer.model_max_length)

    # replace everything before the response with -100 not to calculate loss from these
    if not train_on_source:
        label = [[-100] * len(i) + t for i, t in zip(input["input_ids"], target["input_ids"])]
    else:
        label = [i + t for i, t in zip(input["input_ids"], target["input_ids"])]

    input = {k: [i + t for i, t in zip(input[k], target[k])] for k in input}
    input["text"] = input_text
    input["labels"] = label

    return input


print(f"train len before: {len(train_dataset)}")
print(f"eval len before: {len(eval_dataset)}")

train_dataset = train_dataset.filter(filtering_function, batched=True)
eval_dataset = eval_dataset.filter(filtering_function, batched=True)

print(f"train len after: {len(train_dataset)}")
print(f"eval len after: {len(eval_dataset)}")

train_dataset = train_dataset.map(
    partial(
        preprocess_function,
        tokenizer=tokenizer,
        sep=sep,
        train_on_source=train_on_source
    ),
    batched=True,
    remove_columns=["system_prompt", "question", "response"]
)
eval_dataset = eval_dataset.map(
    partial(
        preprocess_function,
        tokenizer=tokenizer,
        sep=sep
    ),
    batched=True,
    remove_columns=["system_prompt", "question", "response"]
)
train_dataset = train_dataset.train_test_split(0.15)

training_args = TrainingArguments(
    output_dir=output_dir,
    # auto_find_batch_size=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=50,
    bf16=True,
    save_total_limit=2,
    optim=optim
)

trainer = Trainer(
    model,
    train_dataset=train_dataset["train"],
    data_collator=collator,
    eval_dataset=train_dataset["test"],
    args=training_args
)

trainer.train()

# trainer.model.save_pretrained("open_orca_qlora")
# tokenizer.save_pretrained("open_orca_qlora")

evaluate_model(
    trainer,
    tokenizer=tokenizer,
    dataset=eval_dataset,
    output_fname=output_fname
)





