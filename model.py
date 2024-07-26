from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

def load_model(model_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.config.max_length = 54752
    return model, tokenizer
