from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import torch
import huggingface_hub
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "EE_Ion_en_es-v1_1_alpha-GPTQ"

print(f"Loading base model: {model_name_or_path}")

# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     offload_folder="."
# )

# huggingface_hub.login()

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                           model_basename=model_name_or_path,
                                           use_safetensors=True,
                                           trust_remote_code=True,
                                           device="cuda:0",
                                           use_triton=False,
                                           quantize_config=None)

model.push_to_hub("iongpt/EE_Ion_en_es-v1_1_alpha-GPTQ", use_temp_dir=False)
