import os
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import get_res, load_json, save_json, normalize_text
from vectors import (
    load_demographic_vectors_for_agent,
    load_value_vector,
    build_agent_answers,
)
from prompting import build_question_text, system_prompt_map
from cli import is_demographic_method, is_value_vector_method

random.seed(42)
model_cache = {}
LANGS = ["en", "zh", "es", "ar", "ru"]
profile_prompt_map = {
    "en": "Please answer based on the following personal profile:",
    "zh": "请根据以下个人资料作答：",
    "ar": "يرجى الإجابة بناءً على الملف الشخصي التالي:",
    "es": "Por favor, responde según el siguiente perfil personal:",
    "ru": "Пожалуйста, ответьте, основываясь на следующем личном профиле:",
}


def build_output_path(args, q_codes_str: str) -> str:
    def p(name, val):
        return f"{name}_{val}" if val is not None else f"{name}_None"

    method_name = args.method
    if args.use_stories:
        method_name = method_name.replace("prompt_engineering", "background_story")
    parts = [
        args.output_dir,
        args.model_name,
        f"num_agents_{args.num_agents}",
        method_name,
        p("temp", args.temperature),
        f"qcodes_{q_codes_str or 'none'}",
        p("coef", args.coef),
        p("noise", args.noise_std),
        p("steer", args.steering_type),
    ]
    return os.path.join(*parts)


def call_model(agent_profile_full, question_text, args, vector, value_vector, hidden_states_save_dir, agent_id, qid):
    optional_args = {
        k: getattr(args, k)
        for k in ["coef", "steering_type", "noise_std", "max_new_tokens"]
        if getattr(args, k, None) is not None
    }
    return get_res(
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


def parse_numeric_output(text: str):
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return int(float(match.group()))
    except ValueError:
        return None


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

    questions = questions_map[lang]
    agent_profile = agents[lang][agent_id]
    system_prompt = system_prompt_map[lang]
    if agent_profile != "":
        agent_profile = profile_prompt_map[lang] + "\n" + agent_profile

    agent_profile_full = system_prompt + "\n" + agent_profile
    failed_qids = failed_qids or list(questions.keys())

    vector = None
    if is_demographic_method(args.method):
        agents_data = load_json(
            f"agent_profile/demographic/wvs_demographic_{args.num_agents}.json"
        )
        answers = build_agent_answers(agent_id, agents_data, args.demographic_vectors_dir)
        vector = load_demographic_vectors_for_agent(
            answers, args.demographic_vectors_dir, args.vector_q_codes
        )

    value_vector = None
    if is_value_vector_method(args.method):
        value_vector = load_value_vector(args.value_vectors_dir)

    for qid in tqdm(failed_qids, desc=f"Agent {agent_id} questions ({lang})", unit="q"):
        qinfo = questions[qid]
        question_text = build_question_text(qinfo, lang, args.method)

        result = None
        raw_output = None
        option_text_to_num = {}
        options = qinfo.get("options", {})
        if options:
            option_text_to_num = {normalize_text(v): int(k) for k, v in options.items()}
        for attempt in range(args.max_attempts):
            try:
                res = call_model(
                    agent_profile_full, question_text, args, vector, value_vector, hidden_states_save_dir, agent_id, qid
                )
                raw_output = res["output"].strip()
                parse = False
                parsed = parse_numeric_output(raw_output)
                if parsed is not None:
                    result = parsed
                    parse = True
                if option_text_to_num:
                    normalized_output = normalize_text(raw_output)
                    for opt_text, opt_num in option_text_to_num.items():
                        if opt_text in normalized_output:
                            result = opt_num
                            parse = True
                            break
                if parse:
                    break
            except Exception as e:
                print(
                    f"[ERROR] agent {agent_id}, question {qid}, attempt {attempt}: {e}"
                )
        if result is None:
            result = "FAILED_TO_PARSE_NUMBER"
            print(
                f"[WARNING] agent {agent_id}, question {qid}: "
                f"cannot parse output -> {raw_output}"
            )

        existing_data[qid] = {
            "input": res.get("input", ""),
            "output": raw_output,
            "result": result,
        }

    existing_data["lang"] = lang
    os.makedirs(output_path, exist_ok=True)
    save_json(existing_data, os.path.join(output_path, f"{agent_id}.json"))


def process_agents(agents, questions_map, args):
    if not hasattr(args, "vector_q_codes") or args.vector_q_codes is None:
        q_codes_str = "None"
    else:
        q_codes_str = "_".join(f"{q}-{l}" for q, l in args.vector_q_codes.items())
    output_path = build_output_path(args, q_codes_str)
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
                existing_data.get("lang", None),
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
