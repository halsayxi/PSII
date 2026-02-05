import json
import os
import time
from tqdm import tqdm
from utils import get_api_res
from concurrent.futures import ThreadPoolExecutor, as_completed

# default_template = (
#     "Please generate a coherent and engaging story about a person with the following profile information:\n\n"
#     "{profile_text}\n\n"
#     "Create a narrative that captures this person's background, values, opinions, and life experiences. "
#     "Include their thought processes, beliefs, emotional reactions, relationships, behavioral patterns, "
#     "and how they typically act in various situations. "
#     "The story should be in the first person and between 200-300 words."
# )

default_template = (
    "You are to write a **realistic, engaging, and coherent story** about a person using the profile information below. "
    "The story will be used as a simulated human profile for social research or surveys.\n\n"
    "Profile information:\n{profile_text}\n\n"
    "Requirements for the story:\n"
    "- Use **first person** narrative.\n"
    "- Include the person's **background, upbringing, and key life events**.\n"
    "- Show their **values, beliefs, and opinions**.\n"
    "- Include **emotional reactions, thought processes, relationships, and behavioral patterns**.\n"
    "- Illustrate how they typically act in various situations.\n"
    "- Make the story **diverse and realistic**, reflecting a unique life trajectory.\n"
    "- Use natural language; the story should feel like it was **written by the person themselves**.\n"
    "- Length: **400–600 words**.\n"
    "- **Do not include any explanations, prefixes, or post-text comments**. Only output the story."
)


def generate_one_story(
    uid, profile_text, model_name, temperature, max_retries=3, retry_delay=1
):
    prompt = default_template.format(profile_text=profile_text)
    for attempt in range(1, max_retries + 1):
        try:
            story = get_api_res(
                role="You are a professional storyteller.",
                exp=prompt,
                model_name=model_name,
                temperature=temperature,
            )
            return uid, story.strip()
        except Exception as e:
            print(f"[Warning] UID {uid}, attempt {attempt} failed: {e}")
            time.sleep(retry_delay)

    print(f"[Error] UID {uid} failed after {max_retries} attempts. Skipping this UID.")
    return None


def generate_stories(
    input_json_path,
    output_json_path,
    model_name="gpt-4o-mini",
    temperature=0.7,
    max_workers=8,
    max_retries=3,
    retry_delay=1,
):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            stories = json.load(f)
        print(f"Loaded existing stories: {len(stories)} / {len(data)}")
    else:
        stories = {}

    uids_to_generate = [uid for uid in data.keys() if uid not in stories]
    print(f"UIDs to generate: {len(uids_to_generate)}")

    if not uids_to_generate:
        print("All stories already generated. Nothing to do.")
        return

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for uid in uids_to_generate:
            futures.append(
                executor.submit(
                    generate_one_story,
                    uid,
                    data[uid],
                    model_name,
                    temperature,
                    max_retries,
                    retry_delay,
                )
            )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating stories"
        ):
            try:
                result = future.result()
                if result is not None:
                    uid, story = result
                    stories[uid] = story
            except Exception as e:
                print(f"[Critical Error] Unexpected failure: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stories, f, ensure_ascii=False, indent=2)

    print(f"Saved stories to: {output_json_path}. Total stories: {len(stories)}")


if __name__ == "__main__":
    generate_stories(
        input_json_path="agent_profile/descriptions/wvs_demographic_descriptions_100.json",
        output_json_path="agent_profile/stories/wvs_stories_100_strengthened.json",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_workers=1,
        max_retries=3,
        retry_delay=1,
    )
