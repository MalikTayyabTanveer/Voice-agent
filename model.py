from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

def load_model(model_id, gpu_memory_utilization, max_model_len):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    # Adjust GPU memory utilization if supported (this line is just a placeholder and needs actual implementation if supported by the library)
    model.config.gpu_memory_utilization = gpu_memory_utilization

    # Set max sequence length
    model.config.max_length = max_model_len
    return model, tokenizer
