import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
)
from torch.optim import AdamW
from tqdm import tqdm
import argparse
from utils import get_model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train single-language embeddings (PoLM style)"
    )
    parser.add_argument(
        "--lang", type=str, required=True, help="Language code, e.g. es, fr, de"
    )
    parser.add_argument(
        "--n_samples", type=int, default=20000, help="Number of text samples to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--sigma_noise",
        type=float,
        default=0.01,
        help="Standard deviation of noise added to language embedding",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run model on"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_length", type=int, default=64, help="Maximum token length for tokenizer"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Scaling factor for language embedding"
    )
    parser.add_argument(
        "--output_dir", type=str, default="value_vectors/vectors_qwen2.5-7b", help="Output dir"
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen2.5-7b", help="Model name"
    )
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def load_texts(lang: str, n_samples: int):
    parquet_path = f"value_vectors/data/culturax/{lang}/{lang}_part_00000.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    dataset = load_dataset("parquet", data_files=parquet_path)["train"]

    texts = []
    for i, example in enumerate(
        tqdm(dataset, total=n_samples, desc=f"Loading {lang} data")
    ):
        if i >= n_samples:
            break
        texts.append(example["text"])

    return texts


def prepare_dataloader(texts, model_name, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        fix_mistral_regex=True,
    )
    tokenizer.pad_token = tokenizer.eos_token  # Avoid pad token errors

    def tokenize_fn(text):
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    tokenized_texts = [tokenize_fn(t) for t in texts]
    dataloader = DataLoader(tokenized_texts, batch_size=batch_size, shuffle=True)
    return tokenizer, dataloader


def initialize_model(model_name, device, lr):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    ).to(device)
    model.eval()  # Freeze the model
    for p in model.parameters():
        p.requires_grad = False

    d_h = model.config.hidden_size
    language_embedding = nn.Parameter(torch.randn(1, d_h, device=device))
    optimizer = AdamW([language_embedding], lr=lr)

    return model, language_embedding, optimizer


def inject_language(h_t, e_lang):
    return h_t + e_lang


def train_language_embedding(args):
    set_seed(args.seed)

    # Load texts
    texts = load_texts(args.lang, args.n_samples)

    # Prepare tokenizer and dataloader
    model_path = get_model_path(args.model_name)
    tokenizer, dataloader = prepare_dataloader(
        texts, model_path, args.batch_size, args.max_length
    )

    # Initialize model and language embedding
    model, language_embedding, optimizer = initialize_model(
        model_path,
        args.device,
        args.lr,
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, desc=f"Training {args.lang} embedding"):
            input_ids = batch["input_ids"].squeeze(1).to(args.device)
            attention_mask = batch["attention_mask"].squeeze(1).to(args.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            h_last = outputs.hidden_states[-1][:, -1, :]

            e_lang = language_embedding[0]
            e_lang_noisy = e_lang + args.sigma_noise * torch.randn_like(e_lang)  # noise
            h_prime = h_last + args.alpha * e_lang_noisy

            logits = model.lm_head(h_prime)
            targets = input_ids[:, -1]
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[{args.lang}] Epoch {epoch+1}/{args.epochs}, Loss = {loss.item():.4f}")

    # Save language embedding
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = f"{args.output_dir}/language_embedding_{args.lang}.pt"
    torch.save(language_embedding.detach().cpu(), save_path)
    print(f"✅ Language embedding for {args.lang} saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    train_language_embedding(args)
