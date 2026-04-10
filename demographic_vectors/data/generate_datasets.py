import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import httpx


FEATURE_DATA = {
    "Q260": {
        "question_text": "Respondent's Sex",
        "options": {"1": "Male", "2": "Female"},
    },
    "Q263": {
        "question_text": "Immigrant Status",
        "options": {"1": "Born in this country", "2": "Immigrant to this country"},
    },
    "Q271": {
        "question_text": "Living with Parents/In-laws",
        "options": {
            "1": "No",
            "2": "Yes, own parent(s)",
            "3": "Yes, parent(s)-in-law",
            "4": "Yes, both own parent(s) and parent(s)-in-law",
        },
    },
    "Q273": {
        "question_text": "Marital Status",
        "options": {
            "1": "Married",
            "2": "Living together as married",
            "3": "Divorced",
            "4": "Separated",
            "5": "Widowed",
            "6": "Single",
        },
    },
    "Q275": {
        "question_text": "Highest Educational Level",
        "options": {
            "0": "Early childhood education (ISCED 0) / no education",
            "1": "Primary education (ISCED 1)",
            "2": "Lower secondary education (ISCED 2)",
            "3": "Upper secondary education (ISCED 3)",
            "4": "Post-secondary non-tertiary education (ISCED 4)",
            "5": "Short-cycle tertiary education (ISCED 5)",
            "6": "Bachelor or equivalent (ISCED 6)",
            "7": "Master or equivalent (ISCED 7)",
            "8": "Doctoral or equivalent (ISCED 8)",
        },
    },
    "Q279": {
        "question_text": "Employment Status and Hours",
        "options": {
            "1": "Full time employee (30 hours a week or more)",
            "2": "Part time employee (less than 30 hours a week)",
            "3": "Self-employed",
            "4": "Retired/pensioned",
            "5": "Housewife not otherwise employed",
            "6": "Student",
            "7": "Unemployed",
            "8": "Other",
        },
    },
    "Q281": {
        "question_text": "Occupational Group",
        "options": {
            "1": "Professional and technical",
            "2": "Higher administrative",
            "3": "Clerical",
            "4": "Sales",
            "5": "Service",
            "6": "Skilled worker",
            "7": "Semi-skilled worker",
            "8": "Unskilled worker",
            "9": "Farm worker",
            "10": "Farm proprietor, farm manager",
            "0": "Never had a job",
        },
    },
    "Q284": {
        "question_text": "Employer Type (Sector)",
        "options": {
            "1": "Government or public institution",
            "2": "Private business or industry",
            "3": "Private non-profit organization",
        },
    },
    "Q285": {
        "question_text": "Chief Wage Earner Status",
        "options": {"1": "Yes", "2": "No"},
    },
    "Q286": {
        "question_text": "Family Financial Situation (Savings)",
        "options": {
            "1": "Saved money",
            "2": "Just get by",
            "3": "Spent some savings",
            "4": "Spent savings and borrowed money",
        },
    },
    "Q287": {
        "question_text": "Subjective Social Class",
        "options": {
            "1": "Upper class",
            "2": "Upper middle class",
            "3": "Lower middle class",
            "4": "Working class",
            "5": "Lower class",
        },
    },
    "Q288": {
        "question_text": "Household Income Group (1-10 Scale)",
        "options": {
            "1": "Lowest group",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "10": "Highest group",
        },
    },
    "Q289": {
        "question_text": "Religious Denomination",
        "options": {
            "0": "No: do not belong to a denomination",
            "1": "Roman Catholic",
            "2": "Protestant",
            "3": "Orthodox (Russian/Greek/etc.)",
            "4": "Jew",
            "5": "Muslim",
            "6": "Hindu",
            "7": "Buddhist",
            "8": "Other (write in)",
        },
    },
    "Q290": {
        "question_text": "Respondent's Ethnic Group",
        "options": {
            "1": "White",
            "2": "Black",
            "3": "South Asian Indian, Pakistani, etc.",
            "4": "East Asian Chinese, Japanese, etc.",
            "5": "Arabic, Central Asian",
            "6": "Other (write in)",
        },
    },
}

API_KEY = ""
BASE_URL = ""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt-4o")
    parser.add_argument("--output-dir", type=str, default="demographic_data")
    parser.add_argument("--cache-dir", type=str, default="question_cache")
    parser.add_argument("--prompts-template", type=str, default="prompts_pollsim")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def get_openai_client():
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(
            base_url=BASE_URL,
            follow_redirects=True,
        ),
    )


def generate_with_api(
    prompt_text,
    system_role,
    temperature,
    client,
    model_name,
    seed=None,
):
    try:
        request_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt_text},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }

        if seed is not None:
            request_kwargs["seed"] = seed

        response = client.chat.completions.create(**request_kwargs)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {e}")
        return None


def generate_and_cache_questions(
    q_code,
    feature_category,
    client,
    cache_dir,
    model_name,
    questions_prompt,
    seed=None,
):
    cache_name = f"{q_code}_questions.json" if seed is None else f"{q_code}_questions_seed{seed}.json"
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["questions"]

    print(
        f"\n--- Stage 1.1: Generating 40 Questions for Category: {feature_category} ({q_code}) ---"
    )

    prompt_text = questions_prompt.format(FEATURE_CATEGORY=feature_category)
    system_role = "You are a professional social researcher. Your task is to generate 40 survey questions relevant to the given category and return a JSON object."

    generated_data = generate_with_api(
        prompt_text=prompt_text,
        system_role=system_role,
        temperature=0.5,
        client=client,
        model_name=model_name,
        seed=seed,
    )

    if generated_data and "questions" in generated_data:
        questions = generated_data["questions"]
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"questions": questions}, f, ensure_ascii=False, indent=4)
        print(f"Successfully cached questions for {q_code}.")
        return questions
    else:
        print(f"Failed to generate questions for {q_code}.")
        return None


def generate_demographic_instructions(
    feature_category,
    feature_value,
    client,
    model_name,
    instructions_prompt,
    seed=None,
):
    prompt_text = instructions_prompt.format(
        FEATURE_CATEGORY=feature_category, FEATURE_VALUE=feature_value
    )
    system_role = "You are a professional instruction designer. Your task is to strictly follow the instructions and output only a single JSON object."

    generated_data = generate_with_api(
        prompt_text=prompt_text,
        system_role=system_role,
        temperature=0.7,
        client=client,
        model_name=model_name,
        seed=seed,
    )
    if generated_data and "instructions" in generated_data:
        return generated_data["instructions"]
    return None


def main():
    args = parse_args()

    model_name = args.model_name
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    seed = args.seed
    prompts_template = args.prompts_template

    import importlib
    module = importlib.import_module(f".{prompts_template}", package=__package__)
    PROMPTS = module.PROMPTS
    QUESTIONS_PROMPT = PROMPTS["generate_questions"]
    INSTRUCTIONS_PROMPT = PROMPTS["generate_instructions"]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_dir)
    cache_dir = os.path.join(base_dir, cache_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    client = get_openai_client()

    category_questions_cache = {}
    print("\n--- Stage 1: Generating and Caching Questions per Category ---")

    for q_code, config in tqdm(FEATURE_DATA.items(), desc="Generating Questions"):
        final_path = os.path.join(output_dir, f"{q_code}.json")
        if os.path.exists(final_path):
            print(f"Skip {q_code}: {final_path} already exists.")
            continue

        questions = generate_and_cache_questions(
            q_code=q_code,
            feature_category=config["question_text"],
            client=client,
            cache_dir=cache_dir,
            model_name=model_name,
            questions_prompt=QUESTIONS_PROMPT,
            seed=seed,
        )
        if questions:
            category_questions_cache[q_code] = questions

    aggregated_data = {
        q_code: {
            "feature_category": FEATURE_DATA[q_code]["question_text"],
            "instructions": {},
        }
        for q_code in FEATURE_DATA.keys()
    }

    all_features = []
    for q_code, config in FEATURE_DATA.items():
        for option_code, feature_value in config["options"].items():
            all_features.append(
                (q_code, option_code, feature_value, config["question_text"])
            )

    print("\n--- Stage 2: Generating Instructions and Aggregating Data ---")

    for q_code, option_code, feature_value, feature_category in tqdm(
        all_features, desc="Processing All Feature Values"
    ):

        final_path = os.path.join(output_dir, f"{q_code}.json")
        if os.path.exists(final_path):
            print(f"Skip saving {q_code}: file already exists.")
            continue

        instructions = generate_demographic_instructions(
            feature_category=feature_category,
            feature_value=feature_value,
            client=client,
            model_name=model_name,
            instructions_prompt=INSTRUCTIONS_PROMPT,
            seed=seed,
        )

        if instructions:
            aggregated_data[q_code]["instructions"][feature_value] = {
                "option_code": option_code,
                "instructions": instructions,
            }

    print("\n--- Stage 3: Finalizing and Saving Aggregated Files ---")

    for q_code, data in tqdm(aggregated_data.items(), desc="Saving Files"):
        final_path = os.path.join(output_dir, f"{q_code}.json")
        if os.path.exists(final_path):
            print(f"Skip saving {q_code}: file already exists.")
            continue
        
        if q_code not in category_questions_cache:
            continue

        final_data = {
            "feature_category": data["feature_category"],
            "q_code": q_code,
            "questions": category_questions_cache[q_code],
            "instructions": data["instructions"],
        }

        final_path = os.path.join(output_dir, f"{q_code}.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"Saved aggregated file: {final_path}")


if __name__ == "__main__":
    main()