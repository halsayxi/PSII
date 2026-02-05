import json
import time
from tqdm import tqdm
from utils import get_api_res
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import string


def translate_one(
    uid,
    en_text,
    lang_code,
    lang_name,
    model_name,
    temperature,
    max_retries=3,
    retry_delay=1,
):
    system_role = "You are a professional translator."
    user_prompt = (
        f"Translate the following text into {lang_name}. "
        f"Keep the meaning accurate and the style neutral.\n\n"
        f"Only return the translated text, no prefixes or explanations.\n\n"
        f"Text:\n{en_text}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            translated = get_api_res(
                role=system_role,
                exp=user_prompt,
                model_name=model_name,
                temperature=temperature,
            )
            return uid, lang_code, translated.strip()
        except Exception as e:
            print(
                f"[Warning] UID {uid}, lang {lang_code}, attempt {attempt} failed: {e}"
            )
            time.sleep(retry_delay)

    print(
        f"[Error] UID {uid}, lang {lang_code} failed after {max_retries} attempts. Using original text."
    )
    return uid, lang_code, en_text


def normalize_text(s):
    return s.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def translate_one_survey(
    uid,
    en_text,
    lang_code,
    lang_name,
    model_name,
    temperature,
    max_retries=3,
    retry_delay=1,
):
    if isinstance(en_text, str) and en_text.strip().isdigit():
        return uid, lang_code, en_text.strip()

    system_role = "You are a professional translator."
    user_prompt = (
        f"Translate the following text into {lang_name}.\n"
        f"The text may be a question or an answer option. "
        f"Keep the meaning accurate and style neutral.\n"
        f"Only return the translated text, no prefixes or explanations.\n\n"
        f"Text:\n{en_text}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            translated = get_api_res(
                role=system_role,
                exp=user_prompt,
                model_name=model_name,
                temperature=temperature,
            )
            return uid, lang_code, translated.strip()
        except Exception as e:
            print(
                f"[Warning] UID {uid}, lang {lang_code}, attempt {attempt} failed: {e}"
            )
            time.sleep(retry_delay)

    print(
        f"[Error] UID {uid}, lang {lang_code} failed after {max_retries} attempts. Using original text."
    )
    return uid, lang_code, en_text


def translate_survey_json(
    input_json_path,
    output_prefix,
    model_name="gpt-3.5-turbo-0125",
    temperature=0.2,
    max_workers=8,
    max_retries=3,
    retry_delay=1,
):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_langs = {
        "zh": "Chinese (Simplified)",
        "ar": "Arabic",
        "es": "Spanish",
        "ru": "Russian",
    }

    for lang_code, lang_name in target_langs.items():
        output_path = f"{output_prefix}_{lang_code}.json"
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                translated_data = json.load(f)
            print(f"[Info] Loaded existing translations for {lang_code}")
        else:
            translated_data = {}

        tasks = []
        for qid, q_content in data.items():
            if qid not in translated_data:
                translated_data[qid] = {}

            if "question_text" in q_content:
                original_text = q_content["question_text"]
                if "question_text" not in translated_data[qid]:
                    tasks.append((qid, "question_text", original_text))

            if "options" in q_content:
                translated_data[qid].setdefault("options", {})
                for opt_id, opt_text in q_content["options"].items():
                    if opt_id not in translated_data[qid]["options"]:
                        tasks.append((qid, f"options.{opt_id}", opt_text))

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for uid, key, text in tasks:
                futures.append(
                    executor.submit(
                        translate_one_survey,
                        uid=f"{uid}|{key}",
                        en_text=text,
                        lang_code=lang_code,
                        lang_name=lang_name,
                        model_name=model_name,
                        temperature=temperature,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                    )
                )

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Translating {lang_code}",
            ):
                uid_key, _, translated_text = future.result()
                uid, key = uid_key.split("|")
                if key.startswith("options."):
                    opt_id = key.split(".")[1]
                    translated_data[uid].setdefault("options", {})[
                        opt_id
                    ] = translated_text
                else:
                    translated_data[uid][key] = translated_text

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {output_path}")


def translate_profiles(
    input_json_path,
    output_prefix,
    model_name="gpt-3.5-turbo-0125",
    temperature=0.2,
    max_workers=8,
    max_retries=3,
    retry_delay=1,
):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_langs = {
        "zh": "Chinese (Simplified)",
        "ar": "Arabic",
        "es": "Spanish",
        "ru": "Russian",
    }

    results_by_lang = {}
    to_translate = {}

    for lang_code in target_langs:
        output_path = f"{output_prefix}_{lang_code}.json"
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(
                f"[Info] Loaded existing translations for {lang_code}, {len(existing)} entries"
            )
        else:
            existing = {}
        results_by_lang[lang_code] = existing

        missing = {uid: text for uid, text in data.items() if uid not in existing}
        to_translate[lang_code] = missing
        print(f"[Info] {lang_code}: {len(missing)} entries to translate")

    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for lang_code, missing_items in to_translate.items():
            lang_name = target_langs[lang_code]
            for uid, en_text in missing_items.items():
                futures.append(
                    executor.submit(
                        translate_one,
                        uid,
                        en_text,
                        lang_code,
                        lang_name,
                        model_name,
                        temperature,
                        max_retries,
                        retry_delay,
                    )
                )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Translating"
        ):
            try:
                uid, lang_code, translated = future.result()
                results_by_lang[lang_code][uid] = translated
            except Exception as e:
                print(f"[Critical Error] Translation failed unexpectedly: {e}")

    for lang_code in target_langs:
        output_path = f"{output_prefix}_{lang_code}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_by_lang[lang_code], f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    translate_profiles(
        input_json_path="agent_profile/descriptions/wvs_demographic_descriptions_500.json",
        output_prefix="agent_profile/descriptions/wvs_demographic_descriptions_500",
    )
