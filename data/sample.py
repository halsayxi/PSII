import pandas as pd


def random_sample_csv(input_file, sample_size=100):
    df = pd.read_csv(input_file, low_memory=False)

    df.insert(0, "original_id", df.index + 1)

    if len(df) <= sample_size:
        print(f"Warning: the file has only {len(df)} rows, all rows will be exported")
        sampled_df = df.copy()
        actual_size = len(df)
    else:
        sampled_df = df.sample(n=sample_size, random_state=42).copy()
        actual_size = sample_size

    sampled_df.insert(0, "id", range(1, actual_size + 1))

    if input_file.endswith(".csv"):
        output_file = input_file.replace(".csv", f"_{sample_size}.csv")
    else:
        output_file = f"{input_file}_{sample_size}.csv"

    sampled_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    return output_file


if __name__ == "__main__":
    input_csv = "WVS_Cross-National_Wave_7_csv_v6_0.csv"
    random_sample_csv(input_csv, 500)
