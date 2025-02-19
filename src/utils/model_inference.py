from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.generation.utils import GenerationConfig
import torch
from utils.utils import extract_abcd

def model_load(address, model_type):
    if model_type == "qwen":
        return qwen_load(address)
    if model_type == "qwen25":
        return qwen25_load(address)
    if model_type == "glm4":
        return glm4_load(address)
    if model_type == "llama":
        return llama_load(address)

def model_inference(model_type, instruction,model,tokenizer):
    if model_type == "qwen":
        return qwen_inference(instruction,model,tokenizer)
    if model_type == "qwen25":
        return qwen25_inference(instruction,model,tokenizer)
    if model_type == "glm4":
        return glm4_inference(instruction,model,tokenizer)
    if model_type == "llama":
        return llama_inference(instruction,model,tokenizer)


def qwen_load(address):
    tokenizer = AutoTokenizer.from_pretrained(address, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(address, torch_dtype="auto", device_map="auto",
                                                 trust_remote_code=True)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to("cuda")
    model.eval()
    return model,tokenizer

def qwen25_load(address):
    model = AutoModelForCausalLM.from_pretrained(
        address,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(address)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to("cuda")
    model.eval()
    return model,tokenizer



def glm4_load(address):
    tokenizer = AutoTokenizer.from_pretrained(address, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        address,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda").eval()
    return model, tokenizer


def llama_load(address):
    tokenizer = AutoTokenizer.from_pretrained(address)
    model = AutoModelForCausalLM.from_pretrained(
        address,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def qwen_inference(instruction,model,tokenizer):
    inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to('cuda')
    generate_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(instruction):]
    return outputs

def qwen25_inference(instruction, model, tokenizer):

    prompt = instruction
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def glm4_inference(instruction, model, tokenizer):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": instruction}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )

    inputs = inputs.to("cuda")
    gen_kwargs = {"do_sample": True, "top_k": 1}

    with torch.no_grad():

        # Generate the output
        generated = model.generate(**inputs, **gen_kwargs)
        outputs = tokenizer.decode(generated[:, inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)

        return outputs


def llama_inference(instruction,model,tokenizer):
    messages = [
        {"role": "user", "content": instruction},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    outputs = tokenizer.decode(response, skip_special_tokens=True)
    return outputs

