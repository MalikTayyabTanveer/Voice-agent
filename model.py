from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import bitsandbytes as bnb

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map='auto',  # Automatically places layers on the available devices
)
    # Adjust GPU memory utilization if supported (this line is just a placeholder and needs actual implementation if supported by the library)

    # Set max sequence length
    model.config.max_length = 54752
    return model, tokenizer
