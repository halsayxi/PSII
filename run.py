import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import get_res, load_json, save_json, normalize_text
import traceback
import random
import re
import torch
from agent_profile.generate_profile import column_map

random.seed(42)
LANGS = ["en", "zh", "es", "ar", "ru"]

system_prompt_map = {
    "en": "Forget you are an AI model. Simulate a human being. Please answer the following question truthfully.",
    "zh": "请忘记你是一个人工智能模型。请模拟人类。请如实回答以下问题。",
    "ar": "انسَ أنك نموذج ذكاء اصطناعي. قم بمحاكاة إنسان. يرجى الإجابة على السؤال التالي بصدق.",
    "es": "Olvida que eres un modelo de IA. Simula ser un ser humano. Por favor responde la siguiente pregunta con sinceridad.",
    "ru": "Забудьте, что вы модель ИИ. Смоделируйте человека. Пожалуйста, честно ответьте на следующий вопрос.",
}

number_prompt_map = {
    "en": "\nPlease ONLY output a number. Do NOT write any words, symbols, punctuation, or explanations.",
    "zh": "\n请仅输出数字。不要写任何文字、符号、标点或解释。",
    "ar": "\nيرجى إخراج رقم فقط. لا تكتب أي كلمات أو رموز أو علامات ترقيم أو تفسيرات.",
    "es": "\nPor favor, muestra SOLO un número. No escribas palabras, símbolos ni explicaciones.",
    "ru": "\nПожалуйста, выведите ТОЛЬКО число. Не пишите никаких слов, символов или объяснений.",
}

profile_prompt_map = {
    "en": "Please answer based on the following personal profile:",
    "zh": "请根据以下个人资料作答：",
    "ar": "يرجى الإجابة بناءً على الملف الشخصي التالي:",
    "es": "Por favor, responde según el siguiente perfil personal:",
    "ru": "Пожалуйста, ответьте, основываясь на следующем личном профиле:",
}

label_map = {
    "en": {"question": "Question", "options": "Options"},
    "zh": {"question": "问题", "options": "选项"},
    "ar": {"question": "السؤال", "options": "الخيارات"},
    "es": {"question": "Pregunta", "options": "Opciones"},
    "ru": {"question": "Вопрос", "options": "Варианты"},
}

model_cache = {}


def parse_vector_q_codes(s):
    """
    Format:
      qcode:layer,qcode:layer,...
      Example: 1:20,2:18,3:14

    Returns:
      dict[qcode] = layer (int)
    """
    result = {}
    for item in s.split(","):
        if ":" not in item:
            raise ValueError(f"Each qcode must have a layer: {item}")
        q, l = item.split(":")
        result[int(q)] = int(l)
    return result


CHOICES = [
    "direct",
    "prompt_engineering",
    "multilingual",
    "requesting_diversity",
    "demographic_vectors",
    "value_vector",
]


def parse_methods(s):
    methods = [m.strip() for m in s.split(",") if m.strip()]
    invalid = [m for m in methods if m not in CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid method(s): {invalid}. Must be one of {CHOICES}"
        )
    sorted_methods = [m for m in CHOICES if m in methods]
    return "_".join(sorted_methods)


def get_argument_parser():
    parser = argparse.ArgumentParser(description="PollSim Arguments")
    parser.add_argument(
        "--model_name", type=str, default="qwen2.5-7b", help="Name of the model to use."
    )
    parser.add_argument(
        "--use_local_model",
        action="store_true",
        help="Whether to use a local model checkpoint instead of API.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum number of tokens to generate per response (for local model).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for model inference.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum number of retry attempts if generation fails.",
    )
    parser.add_argument(
        "--num_agents", type=int, default=100, help="Number of simulated agents."
    )
    parser.add_argument(
        "--coef", type=float, help="Coefficient for steering vector scaling."
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        help="Standard deviation of Gaussian noise added to vectors.",
    )
    parser.add_argument(
        "--steering_type",
        type=str,
        choices=["response", "prompt", "all"],
        help="Type of steering to apply: response, prompt, or all.",
    )
    parser.add_argument(
        "--demographic_vectors_dir",
        type=str,
        default="demographic_vectors/vectors",
        help="Directory containing demographic vectors.",
    )
    parser.add_argument(
        "--value_vectors_dir",
        type=str,
        default="value_vectors/vectors_qwen2.5-7b",
        help="Directory containing value vectors.",
    )
    parser.add_argument(
        "--vector_q_codes",
        type=parse_vector_q_codes,
        help="Vector qcodes with layer, format: qcode:layer,qcode:layer,...",
    )
    parser.add_argument(
        "--use_stories",
        action="store_true",
        help="Whether to use story-based context for agents.",
    )
    parser.add_argument(
        "--method",
        type=parse_methods,
        default="prompt_engineering",
        help="Method(s) to use, comma-separated. Multiple methods will be sorted and joined by underscores.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs."
    )
    parser.add_argument(
        "--save_hidden_states",
        action="store_true",
        help="Whether to save hidden states from the model.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Whether to process agents in reverse order.",
    )
    return parser


def validate_args(args):
    if not args.use_local_model and args.max_new_tokens is not None:
        raise ValueError(
            "--max_new_tokens is only allowed when --use_local_model is set"
        )

    if args.use_stories and "prompt_engineering" not in args.method:
        raise ValueError(
            "--use_stories can only be used with prompt-engineering methods"
        )

    is_demographic_vectors_method = "demographic_vectors" in args.method
    demographic_vector_required = ["coef", "steering_type", "vector_q_codes"]
    non_demographic_vector_forbidden = [
        "coef",
        "steering_type",
        "noise_std",
        "vector_q_codes",
    ]

    if is_demographic_vectors_method:
        for name in demographic_vector_required:
            if getattr(args, name) is None:
                raise ValueError(f"--{name} is required when method='{args.method}'")
    else:
        for name in non_demographic_vector_forbidden:
            if getattr(args, name) is not None:
                raise ValueError(
                    f"--{name} is only allowed for demographic_vector-based methods"
                )


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
    agent_info = None
    for agent in agents_data:
        if int(agent.get("id")) == int(agent_id):
            agent_info = agent
            break
    if agent_info is None:
        return {}

    agent_answers = {}
    available_vectors = [
        f for f in os.listdir(demographic_vectors_dir) if f.endswith(".pt")
    ]
    available_q_codes = set([f.split("_")[0] for f in available_vectors])
    for q_code, column_key in column_map.items():
        value = agent_info.get(column_key, None)
        if value is None:
            continue
        if value < 0:
            continue
        if q_code in available_q_codes:
            agent_answers[q_code] = value
    return agent_answers


def param_dir(name, value):
    return f"{name}_{value}" if value is not None else f"{name}_None"


def process_agents(agents, questions_map, args):
    if not hasattr(args, "vector_q_codes") or args.vector_q_codes is None:
        q_codes_str = "None"
    else:
        q_codes_str = "_".join(f"{q}-{l}" for q, l in args.vector_q_codes.items())
    method_name = args.method
    if args.use_stories:
        method_name = method_name.replace("prompt_engineering", "background_story")
    path_parts = [
        args.output_dir,
        args.model_name,
        f"num_agents_{args.num_agents}",
        method_name,
        f"temp_{args.temperature}",
        f"qcodes_{q_codes_str}",
        param_dir("coef", args.coef),
        param_dir("noise", args.noise_std),
        param_dir("steer", args.steering_type),
    ]
    output_path = os.path.join(*[p for p in path_parts if p is not None])
    os.makedirs(output_path, exist_ok=True)
    hidden_states_save_dir = None
    if args.save_hidden_states:
        hidden_states_save_dir = os.path.join(output_path, "hidden_states")
        os.makedirs(hidden_states_save_dir, exist_ok=True)   

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        if args.reverse:
            agent_range = range(args.num_agents - 1, -1, -1)
        else:
            agent_range = range(args.num_agents)
        for i in agent_range:
            agent_id = str(i + 1)
            agent_file = os.path.join(output_path, f"{agent_id}.json")
            if os.path.exists(agent_file):
                existing_data = load_json(agent_file)
                lang = existing_data.get("lang", None)
                failed_qids = [
                    qid
                    for qid, qres in existing_data.items()
                    if qid != "lang"
                    and isinstance(qres, dict)
                    and qres.get("result") == "FAILED_TO_PARSE_NUMBER"
                ]
                if not failed_qids:
                    print(
                        f"[INFO] {agent_file} all questions done. Skipping agent {agent_id}..."
                    )
                    continue
                else:
                    print(
                        f"[INFO] {agent_file} has {len(failed_qids)} failed questions. Re-asking..."
                    )
            else:
                existing_data = {}
                failed_qids = None
                lang = None

            future = executor.submit(
                process_single_agent,
                agent_id,
                agents,
                failed_qids,
                output_path,
                args,
                existing_data,
                questions_map,
                hidden_states_save_dir,
                lang,
            )
            futures[future] = agent_id

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing agents"
        ):
            agent_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] agent {agent_id}: {type(e).__name__}: {e}")
                print("Traceback:")
                traceback.print_exc()


def process_single_agent(
    agent_id,
    agents,
    failed_qids,
    output_path,
    args,
    existing_data,
    questions_map,
    hidden_states_save_dir,
    lang=None,
):
    if existing_data is None:
        existing_data = {}

    if ("multilingual" in args.method or "value_vector" in args.method) and lang is None:
        lang = random.choice(LANGS)
    elif lang is None:
        lang = "en"

    system_prompt = system_prompt_map[lang]
    questions = questions_map[lang]
    agent_profile = agents[lang][agent_id]
    if agent_profile != "":
        agent_profile = profile_prompt_map[lang] + "\n" + agent_profile

    if failed_qids is None:
        failed_qids = list(questions.keys())

    agent_profile_full = system_prompt + "\n" + agent_profile

    vector = None
    if "demographic_vectors" in args.method:
        agents_data = load_json(
            f"agent_profile/demographic/wvs_demographic_{args.num_agents}.json"
        )
        agent_answers = build_agent_answers(
            agent_id, agents_data, args.demographic_vectors_dir
        )
        vector = load_demographic_vectors_for_agent(
            agent_answers, args.demographic_vectors_dir, args.vector_q_codes
        )

    value_vector = None
    if "value_vector" in args.method:
        value_vector = load_value_vector(args.value_vectors_dir)

    # questions_of_interest = ["Q48", "Q158", "Q112", "Q106"]

    for qid in tqdm(failed_qids, desc=f"Agent {agent_id} questions ({lang})", unit="q"):
        # if qid not in questions_of_interest:
        #     continue
        qinfo = questions[qid]
        question_label = label_map[lang]["question"]
        options_label = label_map[lang]["options"]
        question_text = f"{question_label}: {qinfo.get('question_text', '')}"
        options = qinfo.get("options", {})
        if options:
            opts = [
                f"{k}: {v}" for k, v in sorted(options.items(), key=lambda x: int(x[0]))
            ]
            question_text += f"\n{options_label}:\n" + "\n".join(opts)
        question_text += number_prompt_map[lang]
        if "requesting_diversity" in args.method:
            question_text += "\nPlease try to be as diverse as possible."

        option_text_to_num = {}
        if options:
            option_text_to_num = {normalize_text(v): int(k) for k, v in options.items()}
        attempt = 0
        result = None
        raw_output = None

        while attempt < args.max_attempts:
            attempt += 1
            try:
                optional_args = {}
                for key in [
                    "coef",
                    "steering_type",
                    "noise_std",
                    "max_new_tokens",
                ]:
                    if hasattr(args, key) and getattr(args, key) is not None:
                        optional_args[key] = getattr(args, key)
                res = get_res(
                    agent_profile_full,
                    question_text,
                    args.use_local_model,
                    args.model_name,
                    args.temperature,
                    model_cache,
                    hidden_states_save_dir,
                    agent_id,
                    qid,
                    args.save_hidden_states,
                    vector,
                    value_vector,
                    **optional_args,
                )
                raw_output = res["output"].strip()
                parsed = False

                number_match = re.search(r"[-+]?\d*\.?\d+", raw_output)
                if number_match:
                    numeric_value = number_match.group()
                    try:
                        result = int(float(numeric_value))
                        parsed = True
                    except ValueError:
                        pass
                if not parsed and option_text_to_num:
                    normalized_output = normalize_text(raw_output)
                    for opt_text, opt_num in option_text_to_num.items():
                        if opt_text in normalized_output:
                            result = opt_num
                            parsed = True
                            break
                if parsed:
                    break

            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[ERROR] agent {agent_id}, question {qid}, attempt {attempt}\n{tb}"
                )
                raw_output = f"ERROR:\n{tb}"

        if result is None:
            result = "FAILED_TO_PARSE_NUMBER"
            print(
                f"[WARNING] agent {agent_id}, question {qid}: "
                f"cannot parse output -> {raw_output}"
            )

        final_res = {
            "input": res["input"],
            "output": raw_output,
            "result": result,
        }

        existing_data[qid] = final_res

    existing_data["lang"] = lang

    agent_file = os.path.join(output_path, f"{agent_id}.json")
    save_json(existing_data, agent_file)


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    validate_args(args)

    if "prompt_engineering" in args.method:
        if args.use_stories:
            agents = {
                "en": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}.json"
                ),
                "zh": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_zh.json"
                ),
                "es": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_es.json"
                ),
                "ar": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_ar.json"
                ),
                "ru": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_ru.json"
                ),
            }
        else:
            agents = {
                "en": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}.json"
                ),
                "zh": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_zh.json"
                ),
                "es": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_es.json"
                ),
                "ar": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_ar.json"
                ),
                "ru": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_ru.json"
                ),
            }
    else:
        agents = {
            lang: {str(i + 1): "" for i in range(args.num_agents)} for lang in LANGS
        }

    questions_map = {
        "en": dict(list(load_json("data/questions.json").items())[:259]),
        "zh": dict(list(load_json("data/questions_zh.json").items())[:259]),
        "es": dict(list(load_json("data/questions_es.json").items())[:259]),
        "ar": dict(list(load_json("data/questions_ar.json").items())[:259]),
        "ru": dict(list(load_json("data/questions_ru.json").items())[:259]),
    }

    process_agents(agents, questions_map, args)


if __name__ == "__main__":
    main()
