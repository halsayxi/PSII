from openai import OpenAI
import httpx
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from activation_steer import ActivationSteerer, ActivationSteererMultiple
import string
import numpy as np
import os

API_KEY = ""
BASE_URL = ""


def get_api_res(role, exp, model_name, temperature):
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(
            base_url=BASE_URL,
            follow_redirects=True,
        ),
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": exp},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def get_model_path(model_name):
    if model_name == "qwen2.5-7b":
        return ""
    elif model_name == "qwen2.5-14b":
        return ""
    elif model_name == "mistral-7b":
        return ""
    elif model_name == "mistral-24b":
        return ""
    elif model_name == "llama3.1-8b":
        return ""
    else:
        raise ValueError(f"{model_name} not found")


def get_model(
    model_name,
    max_new_tokens=None,
    temperature=None,
    return_generation_config=True,
    save_hidden_states=False,
):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if return_generation_config:
        if max_new_tokens is None:
            raise ValueError(
                "max_new_tokens is required when return_generation_config=True"
            )
        if temperature is None:
            raise ValueError(
                "temperature is required when return_generation_config=True"
            )
        if isinstance(max_new_tokens, str):
            max_new_tokens = int(max_new_tokens)
        generation_config = {
            "do_sample": True,
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 1e-5),
            "top_p": 0.8,
            "top_k": 20,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if save_hidden_states:
            generation_config["output_hidden_states"] = True
            generation_config["return_dict_in_generate"] = True
        return model, tokenizer, generation_config
    return model, tokenizer


def get_prompt(tokenizer, role, exp):
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": exp},
    ]

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    else:
        inputs = tokenizer(
            "\n".join([m["content"] for m in messages]),
            return_tensors="pt",
            padding=True,
        )

    return inputs


def predict(
    model,
    tokenizer,
    generation_config,
    role,
    exp,
    hidden_states_save_dir,
    agent_id,
    qid,
    save_hidden_states=False,
    vector=None,
    coef=None,
    steering_type=None,
    noise_std=None,
    value_vector=None,
):
    model.eval()
    inputs = get_prompt(tokenizer, role, exp)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_len = input_ids.shape[1]

    saved_hidden = {}

    def save_last_token_hook(module, input, output, layer_idx):
        # output: (batch, seq_len, hidden_dim)
        last_token = output[0, -1].detach().cpu().numpy().astype(np.float16)
        saved_hidden[layer_idx] = last_token

    with torch.no_grad():
        if vector is None and value_vector is not None:
            steerer = ActivationSteerer(
                model,
                vector,
                coeff=coef,
                layer_idx=None,
                positions=steering_type,
                noise_std=noise_std,
                value_vector=value_vector,
            )
        elif vector is not None or value_vector is not None:
            instructions = []
            for vec, layer_idx in vector:
                instructions.append(
                    {
                        "steering_vector": vec,
                        "layer_idx": layer_idx - 1,
                        "positions": steering_type,
                        "noise_std": noise_std,
                        "coeff": coef,
                    }
                )
            steerer = ActivationSteererMultiple(
                model, instructions=instructions, value_vector=value_vector
            )
        else:
            steerer = torch.no_grad()
        
        with steerer:
            if save_hidden_states:
                hooks = []
                for l_idx, layer_module in enumerate(model.model.layers):  
                    hook = layer_module.register_forward_hook(
                        lambda m, i, o, l=l_idx: save_last_token_hook(m, i, o, l)
                    )
                    hooks.append(hook)

                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config,
                )

                for h in hooks:
                    h.remove()
                
                for l_idx, vec in saved_hidden.items():
                    file_name = f"agent_{agent_id}_q_{qid}_layer_{l_idx}.npy"
                    np.save(os.path.join(hidden_states_save_dir, file_name), vec)
            else:   
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config,
                )

            if hasattr(output, "sequences"):
                generated_tokens = output.sequences[:, input_len:]
            else:
                generated_tokens = output[:, input_len:]
            text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    return text


def get_model_and_tokenizer(model_cache, model_name, max_new_tokens, temperature, save_hidden_states=False):
    if model_name not in model_cache:
        model, tokenizer, generation_config = get_model(
            model_name, max_new_tokens, temperature, save_hidden_states=save_hidden_states
        )
        model_cache[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "generation_config": generation_config,
        }
    return model_cache[model_name]


def get_local_res(
    role,
    exp,
    model_name,
    temperature,
    max_new_tokens,
    model_cache,
    hidden_states_save_dir,
    agent_id,
    qid,
    save_hidden_states=False,
    vector=None,
    value_vector=None,
    coef=None,
    steering_type="response",
    noise_std=0.1,
):
    model_data = get_model_and_tokenizer(
        model_cache, model_name, max_new_tokens, temperature, save_hidden_states
    )
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    generation_config = model_data["generation_config"]

    return predict(
        model,
        tokenizer,
        generation_config,
        role,
        exp,
        hidden_states_save_dir,
        agent_id,
        qid,
        save_hidden_states,
        vector=vector,
        coef=coef,
        steering_type=steering_type,
        noise_std=noise_std,
        value_vector=value_vector,
    )


def get_res(
    role,
    exp,
    use_local_model,
    model_name,
    temperature,
    model_cache,
    hidden_states_save_dir,
    agent_id,
    qid,
    save_hidden_states=False,
    vector=None,
    value_vector=None,
    coef=None,
    steering_type=None,
    noise_std=0.0,
    max_new_tokens=5,
):
    role = role.strip()
    exp = exp.strip()
    message = role + "\n" + exp

    if use_local_model:
        res = get_local_res(
            role,
            exp,
            model_name,
            temperature,
            max_new_tokens,
            model_cache,
            hidden_states_save_dir,
            agent_id,
            qid,
            save_hidden_states,
            vector=vector,
            value_vector=value_vector,
            coef=coef,
            steering_type=steering_type,
            noise_std=noise_std,
        )
    else:
        res = get_api_res(role, exp, model_name, temperature)
    message_parts = message.split("\n")
    return {
        "input": [part for part in message_parts],
        "input_continued": message,
        "output": res,
    }


def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"error while loading json file: {file_path}, {str(e)}")
        return None


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(s):
    return s.lower().translate(str.maketrans("", "", string.punctuation)).strip()
