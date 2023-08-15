from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import torch


def load_model_and_tokenizer(model_name_or_path, device):
    print(f"Loading base model: {model_name_or_path}")

    max_memory_mapping = {0: "22GB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        # load_in_4bit=True,
        max_memory=max_memory_mapping,
        torch_dtype=torch.float16,
        offload_folder="."
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer


def generate_translation(model, tokenizer, prompt_template):
    print("\n\n*** Generate:")
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    return tokenizer.decode(output[0])


def use_pipeline_for_translation(model, tokenizer, prompt_template):
    # Prevent printing spurious transformers error when using pipeline
    logging.set_verbosity(logging.CRITICAL)

    print("*** Pipeline:")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    return pipe(prompt_template)[0]['generated_text']


def main():
    model_name_or_path = "iongpt_EE_Ion_en_es-v1_0_alpha-fp16"
    device = "auto"
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, device)

    prompt = "Translate the phrase \"I am sleeping\" from English to Spanish"
    system_message = "As a professional translator with expertise in English and Spanish, you are tasked with translating individual strings from English to Spanish. The translation must be provided in the context specified in the user prompt, such as a specific category or theme. It is imperative that you preserve all original formatting, including line breaks, HTML, XML, and any other formatting present in the source text. Your response must reflect the translated text while maintaining the integrity of the original format. Ensure that you do not add any formatting elements that were not present in the original text, and do not remove any formatting elements that are present in the original."
    prompt_template = f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]
'''

    translation = generate_translation(model, tokenizer, prompt_template)
    print(translation)

    pipeline_translation = use_pipeline_for_translation(model, tokenizer, prompt_template)
    print(pipeline_translation)


if __name__ == "__main__":
    main()
