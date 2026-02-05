import json
import os
from openai import OpenAI
from tqdm import tqdm
import httpx
from prompts_pollsim import PROMPTS


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
MODEL_TO_USE = "gpt-4o"
OUTPUT_DIR = "demographic_data"
CATEGORY_QUESTIONS_CACHE = "question_cache"
QUESTIONS_PROMPT = PROMPTS["generate_questions"]
INSTRUCTIONS_PROMPT = PROMPTS["generate_instructions"]


def get_openai_client():
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(
            base_url=BASE_URL,
            follow_redirects=True,
        ),
    )


def generate_with_api(prompt_text, system_role, temperature, model_name=MODEL_TO_USE):
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt_text},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {e}")
        return None


def generate_and_cache_questions(q_code, feature_category):
    cache_path = os.path.join(CATEGORY_QUESTIONS_CACHE, f"{q_code}_questions.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["questions"]

    print(
        f"\n--- Stage 1.1: Generating 40 Questions for Category: {feature_category} ({q_code}) ---"
    )

    prompt_text = QUESTIONS_PROMPT.format(FEATURE_CATEGORY=feature_category)
    system_role = "You are a professional social researcher. Your task is to generate 40 survey questions relevant to the given category and return a JSON object."

    generated_data = generate_with_api(prompt_text, system_role, 0.5)

    if generated_data and "questions" in generated_data:
        questions = generated_data["questions"]
        os.makedirs(CATEGORY_QUESTIONS_CACHE, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"questions": questions}, f, ensure_ascii=False, indent=4)
        print(f"Successfully cached questions for {q_code}.")
        return questions
    else:
        print(f"Failed to generate questions for {q_code}.")
        return None


def generate_demographic_instructions(feature_category, feature_value):
    prompt_text = INSTRUCTIONS_PROMPT.format(
        FEATURE_CATEGORY=feature_category, FEATURE_VALUE=feature_value
    )
    system_role = "You are a professional instruction designer. Your task is to strictly follow the instructions and output only a single JSON object."

    generated_data = generate_with_api(prompt_text, system_role, 0.7)
    if generated_data and "instructions" in generated_data:
        return generated_data["instructions"]
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CATEGORY_QUESTIONS_CACHE, exist_ok=True)

    category_questions_cache = {}
    print("\n--- Stage 1: Generating and Caching Questions per Category ---")

    for q_code, config in tqdm(FEATURE_DATA.items(), desc="Generating Questions"):
        questions = generate_and_cache_questions(q_code, config["question_text"])
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

        final_path = os.path.join(OUTPUT_DIR, f"{q_code}.json")
        if os.path.exists(final_path):
            with open(final_path, "r", encoding="utf-8") as f:
                temp_data = json.load(f)
            if feature_value in temp_data.get("instructions", {}):
                aggregated_data[q_code]["instructions"] = temp_data["instructions"]
                continue

        instructions = generate_demographic_instructions(feature_category, feature_value)

        if instructions:
            aggregated_data[q_code]["instructions"][feature_value] = {
                "option_code": option_code,
                "instructions": instructions,
            }

    print("\n--- Stage 3: Finalizing and Saving Aggregated Files ---")

    for q_code, data in tqdm(aggregated_data.items(), desc="Saving Files"):
        if q_code not in category_questions_cache:
            continue

        final_data = {
            "feature_category": data["feature_category"],
            "q_code": q_code,
            "questions": category_questions_cache[q_code],
            "instructions": data["instructions"],
        }

        final_path = os.path.join(OUTPUT_DIR, f"{q_code}.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"Saved aggregated file: {final_path}")


if __name__ == "__main__":
    main()
