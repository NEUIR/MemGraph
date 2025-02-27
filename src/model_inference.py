import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any


def model_load(model_path: str, model_type: str) -> Tuple[Any, Any]:
    """
    根据模型类型加载对应的模型和分词器

    Args:
        model_path: 模型文件路径
        model_type: 模型类型，支持"qwen"、"qwen25"、"glm"、"llama"

    Returns:
        包含model和tokenizer的元组
    """
    model_loaders = {
        "qwen": qwen_load,
        "qwen25": qwen25_load,
        "glm": glm_load,
        "llama": llama_load
    }

    if model_type not in model_loaders:
        supported_types = list(model_loaders.keys())
        raise ValueError(f"不支持的模型类型: {model_type}。支持的类型有: {supported_types}")

    return model_loaders[model_type](model_path)


def model_inference(model_type: str, instruction: str, model: Any, tokenizer: Any) -> str:
    """
    使用指定模型进行推理

    Args:
        model_type: 模型类型，支持"qwen"、"qwen25"、"glm"、"llama"
        instruction: 输入文本
        model: 已加载的模型
        tokenizer: 已加载的分词器

    Returns:
        模型生成的输出文本
    """
    inference_functions = {
        "qwen": qwen_inference,
        "qwen25": qwen25_inference,
        "glm": glm_inference,
        "llama": llama_inference
    }

    if model_type not in inference_functions:
        supported_types = list(inference_functions.keys())
        raise ValueError(f"不支持的模型类型: {model_type}。支持的类型有: {supported_types}")

    return inference_functions[model_type](instruction, model, tokenizer)


def qwen_load(model_path: str) -> Tuple[Any, Any]:
    """
    加载Qwen模型和分词器

    Args:
        model_path: 模型文件路径

    Returns:
        包含model和tokenizer的元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to("cuda")
    model.eval()

    return model, tokenizer


def qwen25_load(model_path: str) -> Tuple[Any, Any]:
    """
    加载Qwen25模型和分词器

    Args:
        model_path: 模型文件路径

    Returns:
        包含model和tokenizer的元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to("cuda")
    model.eval()

    return model, tokenizer


def glm_load(model_path: str) -> Tuple[Any, Any]:
    """
    加载GLM模型和分词器

    Args:
        model_path: 模型文件路径

    Returns:
        包含model和tokenizer的元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda").eval()

    return model, tokenizer


def llama_load(model_path: str) -> Tuple[Any, Any]:
    """
    加载LLaMA模型和分词器

    Args:
        model_path: 模型文件路径

    Returns:
        包含model和tokenizer的元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    return model, tokenizer


def qwen_inference(instruction: str, model: Any, tokenizer: Any) -> str:
    """
    使用Qwen模型进行推理

    Args:
        instruction: 输入文本
        model: 已加载的模型
        tokenizer: 已加载的分词器

    Returns:
        模型生成的输出文本
    """
    # 对输入进行编码
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to('cuda')

    # 生成回答
    generate_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 解码并返回结果，跳过输入部分
    outputs = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0][len(instruction):]

    return outputs


def qwen25_inference(instruction: str, model: Any, tokenizer: Any) -> str:
    """
    使用Qwen25模型进行推理

    Args:
        instruction: 输入文本
        model: 已加载的模型
        tokenizer: 已加载的分词器

    Returns:
        模型生成的输出文本
    """
    # 构造系统提示和用户输入
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 对输入进行编码
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回答
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # 提取生成的部分（排除输入部分）
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码并返回结果
    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response


def glm_inference(instruction: str, model: Any, tokenizer: Any) -> str:
    """
    使用GLM模型进行推理

    Args:
        instruction: 输入文本
        model: 已加载的模型
        tokenizer: 已加载的分词器

    Returns:
        模型生成的输出文本
    """
    # 应用聊天模板并编码
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    # 设置生成参数
    gen_kwargs = {"do_sample": True, "top_k": 1}

    # 生成并返回结果
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
        outputs = tokenizer.decode(
            generated[:, inputs['input_ids'].shape[1]:][0],
            skip_special_tokens=True
        )

        return outputs


def llama_inference(instruction: str, model: Any, tokenizer: Any) -> str:
    """
    使用LLaMA模型进行推理

    Args:
        instruction: 输入文本
        model: 已加载的模型
        tokenizer: 已加载的分词器

    Returns:
        模型生成的输出文本
    """
    # 构造消息
    messages = [
        {"role": "user", "content": instruction},
    ]

    # 应用聊天模板并编码
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 设置终止标志
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 生成回答
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # 提取生成的部分（排除输入部分）
    response = outputs[0][input_ids.shape[-1]:]
    outputs = tokenizer.decode(response, skip_special_tokens=True)

    return outputs