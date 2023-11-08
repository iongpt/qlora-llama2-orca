import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    logging
)

# Constants for model configuration
MODEL_CONFIG = {
    "max_memory_mapping": {0: "22GB"},
    "torch_dtype": torch.float16,
    "offload_folder": "."
}

# Constants for text generation
GENERATION_CONFIG = {
    "temperature": 0.7,
    "max_new_tokens": 512,
    "top_p": 0.95,
    "repetition_penalty": 1.15
}

def load_model_and_tokenizer(model_name_or_path: str, device: str = "auto") -> tuple:
    print(f"Loading base model: {model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        **MODEL_CONFIG
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer


def generate_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    print("\n\n*** Generate:")
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, **GENERATION_CONFIG)
    return tokenizer.decode(output[0])


def translate_using_pipeline(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    logging.set_verbosity(logging.CRITICAL)  # Suppressing unnecessary logs

    print("*** Pipeline:")
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **GENERATION_CONFIG
    )

    return text_generator(prompt)[0]['generated_text']


def main():
    model_name_or_path = "iongpt_EE_Ion_en_es-v1_0_alpha-fp16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, device)

    prompt = "Translate the phrase \"I am sleeping\" from English to Spanish"
    system_message = (
        "As a professional translator with expertise in English and Spanish, you are tasked with translating individual "
        "strings from English to Spanish. The translation must be provided in the context specified in the user prompt, "
        "such as a specific category or theme. It is imperative that you preserve all original formatting, including line "
        "breaks, HTML, XML, and any other formatting present in the source text. Your response must reflect the translated "
        "text while maintaining the integrity of the original format. Ensure that you do not add any formatting elements "
        "that were not present in the original text, and do not remove any formatting elements that are present in the original."
    )
    prompt_template = f'[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]\n'

    translation = generate_text(model, tokenizer, prompt_template)
    print(translation)

    pipeline_translation = translate_using_pipeline(model, tokenizer, prompt_template)
    print(pipeline_translation)


if __name__ == "__main__":
    main()
