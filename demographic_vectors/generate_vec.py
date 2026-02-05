import os
import torch
import pandas as pd
from tqdm import tqdm
import argparse
from utils import get_model


@torch.no_grad()
def get_response_hidden_states(
    model,
    tokenizer,
    prompts,
    responses,
):
    num_layers = model.config.num_hidden_layers
    response_avg = [[] for _ in range(num_layers + 1)]
    for prompt, response in tqdm(zip(prompts, responses), total=len(prompts)):
        text = prompt + response
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        outputs = model(**inputs, output_hidden_states=True)
        for layer in range(num_layers + 1):
            h = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1)
            response_avg[layer].append(h.cpu())
        del outputs
    for layer in range(num_layers + 1):
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)

    return response_avg


def process_single_csv(model, tokenizer, csv_path, save_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    question_id = os.path.basename(csv_path).replace(".csv", "")
    value_codes = sorted(df["value_code"].unique())

    print(f"\n===== Processing {question_id} =====")
    print(f"Total samples: {len(df)}")
    print(f"Value codes: {value_codes}")

    all_prompts = df["prompt"].tolist()
    all_responses = df["answer"].tolist()

    all_response_avg = get_response_hidden_states(
        model, tokenizer, all_prompts, all_responses
    )

    all_response_mean = torch.stack(
        [all_response_avg[l].mean(0) for l in range(len(all_response_avg))], dim=0
    )

    for value_code in value_codes:
        save_path = os.path.join(save_dir, f"{question_id}_{value_code}.pt")
        if os.path.exists(save_path):
            print(f"  [SKIP] {save_path} already exists")
            continue

        sub_df = df[df["value_code"] == value_code]
        prompts = sub_df["prompt"].tolist()
        responses = sub_df["answer"].tolist()
        print(f"  value_code={value_code}, samples={len(sub_df)}")
        value_response_avg = get_response_hidden_states(
            model, tokenizer, prompts, responses
        )
        value_response_mean = torch.stack(
            [value_response_avg[l].mean(0) for l in range(len(value_response_avg))],
            dim=0,
        )

        demographic_vector = value_response_mean - all_response_mean

        torch.save(demographic_vector, save_path)
        print(f"    saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b")
    parser.add_argument(
        "--input_dir", type=str, default="demographic_vectors/data/eval_data"
    )
    parser.add_argument("--output_dir", type=str, default="demographic_vectors/vectors")
    args = parser.parse_args()
    model, tokenizer = get_model(
        model_name=args.model_name, return_generation_config=False
    )
    model.eval()
    csv_dir = f"{args.input_dir}/{args.model_name}"
    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        process_single_csv(model, tokenizer, csv_path, args.output_dir)


if __name__ == "__main__":
    main()
