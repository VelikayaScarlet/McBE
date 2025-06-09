import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig

def Loader(model_name):
    if model_name == 'baichuan2':
        model_dir = "path_of_your_model"
    elif model_name == 'qwen2':
        model_dir = "path_of_your_model"
    elif model_name == "internlm2":
        model_dir = "path_of_your_model"    
    else:
        print("There is no "+model_name)
        return
    tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", 
                                trust_remote_code=True, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", 
                                trust_remote_code=True, torch_dtype=torch.float16)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    return tokenizer, model


def baichuan2(prompt, tokenizer, model):
    messages = []
    messages.append({"role": "user", "content": prompt})
    response = model.chat(tokenizer, messages)
    return response

def qwen2(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def internlm2(prompt, tokenizer, model):
    model = model.eval()
    response, history = model.chat(tokenizer, prompt, history=[])
    return response