# This code was copied from here: https://gist.github.com/TheBloke/d31d289d3198c24e0ca68aaf37a19032
# It was included here just for convenience in the process of fine tune, merge and quantize the model.
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os


def main():
    # Hardcoded values
    base_model_name_or_path = "meta-llama/Llama-2-13b-hf"
    peft_model_path = "outputs/"
    output_dir = "merged"
    device = "auto"
    push_to_hub = False

    if device == 'auto':
        device_arg = {'device_map': 'auto'}
    else:
        device_arg = {'device_map': {"": device}}

    print(f"Loading base model: {base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        use_auth_token=True,
        **device_arg
    )

    print(f"Loading PEFT: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path, **device_arg)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    if push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{output_dir}", use_temp_dir=False)
        tokenizer.push_to_hub(f"{output_dir}", use_temp_dir=False)
    else:
        model.save_pretrained(f"{output_dir}")
        tokenizer.save_pretrained(f"{output_dir}")
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
