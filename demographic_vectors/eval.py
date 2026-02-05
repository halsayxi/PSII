import os
import json
import argparse
import pandas as pd
from utils import get_res
from tqdm import tqdm

model_cache = {}


def process_file(json_path, args):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q_code = data["q_code"]
    questions = data.get("questions", [])
    instructions_dict = data.get("instructions", [])

    print(f"\nProcessing q_code: {q_code}")

    rows = []

    with tqdm(
        total=len(instructions_dict), desc=f"q_code {q_code}", leave=True
    ) as qcode_pbar:
        for value_id, value_data in instructions_dict.items():
            instructions_list = value_data.get("instructions", [])
            option_code = value_data.get("option_code", None)
            print(
                f"  Processing value_id: {value_id} ({len(instructions_list)} instructions)"
            )

            with tqdm(
                total=len(instructions_list), desc=f"Value {value_id}", leave=False
            ) as value_pbar:
                for instr_id, instruction in enumerate(instructions_list):
                    print(
                        f"    Processing instruction {instr_id + 1}/{len(instructions_list)}"
                    )

                    with tqdm(
                        total=len(questions), desc=f"Instr {instr_id}", leave=False
                    ) as instr_pbar:
                        for ques_id, question in enumerate(questions):
                            question_id = f"{q_code}_{ques_id}_{value_id}_{instr_id}"
                            answer = get_res(
                                role=instruction,
                                exp=question,
                                model_name=args.model_name,
                                temperature=args.temperature,
                                use_local_model=args.use_local_model,
                                max_new_tokens=args.max_new_tokens,
                                model_cache=model_cache,
                            )

                            rows.append(
                                {
                                    "question_id": question_id,
                                    "value": value_id,
                                    "value_code": option_code,
                                    "instruction": instruction,
                                    "question": question,
                                    "prompt": answer["input_continued"].strip(),
                                    "answer": answer["output"].strip(),
                                }
                            )
                            instr_pbar.update(1)

                    value_pbar.update(1)
            qcode_pbar.update(1)

    json_filename = os.path.basename(json_path)
    csv_filename = os.path.splitext(json_filename)[0] + ".csv"
    save_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, csv_filename)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="demographic_vectors/data/demographic_data"
    )
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--use_local_model", action="store_true")
    parser.add_argument(
        "--output_dir", type=str, default="demographic_vectors/data/eval_data"
    )
    args = parser.parse_args()

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".json") and filename.startswith("Q"):
            json_path = os.path.join(args.input_dir, filename)
            # print(json_path)
            process_file(json_path, args)


if __name__ == "__main__":
    main()
