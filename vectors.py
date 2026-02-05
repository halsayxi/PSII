import os
import torch
from agent_profile.generate_profile import column_map
import random

LANGS = ["en", "zh", "es", "ar", "ru"]


def load_demographic_vectors_for_agent(agent_answers, demographic_vectors_dir, vector_q_codes):
    loaded_vectors = []

    for q_num, layer_idx in vector_q_codes.items():
        if q_num == 9999:
            for q_code, answer in agent_answers.items():
                vector_file = os.path.join(demographic_vectors_dir, f"{q_code}_{answer}.pt")
                if not os.path.exists(vector_file):
                    print(f"Demographic vector file not found for {q_code}: {vector_file}")
                    continue
                try:
                    vector_data = torch.load(vector_file, weights_only=False)
                    if layer_idx == 9999:
                        for l_idx, vector in vector_data.items():
                            loaded_vectors.append((vector, l_idx))
                    else:
                        vector = vector_data[layer_idx]
                        loaded_vectors.append((vector, layer_idx))
                except Exception as e:
                    print(f"Error loading vector for {q_code}: {e}")
            continue

        q_code = f"Q{q_num}"
        if q_code not in agent_answers:
            continue
        answer = agent_answers[q_code]
        vector_file = os.path.join(demographic_vectors_dir, f"{q_code}_{answer}.pt")
        if not os.path.exists(vector_file):
            print(f"Demographic vector file not found for {q_code}: {vector_file}")
            continue
        try:
            vector_data = torch.load(vector_file, weights_only=False)
            if layer_idx == 9999:
                for l_idx, vector in vector_data.items():
                    loaded_vectors.append((vector, l_idx))
            else:
                vector = vector_data[layer_idx]
                loaded_vectors.append((vector, layer_idx))
        except Exception as e:
            print(f"Error loading vector for {q_code}: {e}")
            continue

    return loaded_vectors


def load_value_vector(value_vectors_dir):
    lang = random.choice(LANGS)
    vector_file = os.path.join(value_vectors_dir, f"language_embedding_{lang}.pt")
    if not os.path.exists(vector_file):
        raise FileNotFoundError(
            f"Value vector file not found for {lang}: {vector_file}"
        )
    vector = torch.load(vector_file)
    print(f"Loaded value vector for {lang}, shape = {vector.shape}")
    return vector


def build_agent_answers(agent_id, agents_data, demographic_vectors_dir):
    agent_info = next(
        (a for a in agents_data if int(a.get("id")) == int(agent_id)), None
    )
    if agent_info is None:
        return {}

    agent_answers = {}
    available_vectors = [
        f for f in os.listdir(demographic_vectors_dir) if f.endswith(".pt")
    ]
    available_q_codes = set([f.split("_")[0] for f in available_vectors])
    for q_code, column_key in column_map.items():
        value = agent_info.get(column_key, None)
        if value is None or value < 0:
            continue
        if q_code in available_q_codes:
            agent_answers[q_code] = value
    return agent_answers
