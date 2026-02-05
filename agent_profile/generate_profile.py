import pandas as pd
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(
    BASE, "..", "data", "WVS_Cross-National_Wave_7_csv_v6_0_500.csv"
)
output_json = os.path.join(BASE, "demographic/wvs_demographic_500.json")

column_map = {
    "id": "id",
    "B_COUNTRY": "country",
    "G_TOWNSIZE": "settlement_size",
    "H_SETTLEMENT": "settlement_type",
    "H_URBRURAL": "urban_rural",
    "Q260": "sex",
    "Q261": "birth_year",
    "Q262": "age",
    "Q263": "immigrant_self",
    "Q264": "immigrant_mother",
    "Q265": "immigrant_father",
    "Q266": "birth_country_self",
    "Q267": "birth_country_mother",
    "Q268": "birth_country_father",
    "Q269": "citizen",
    "Q270": "household_size",
    "Q271": "live_with_parents",
    "Q272": "home_language",
    "Q273": "marital_status",
    "Q274": "have_children",
    "Q275": "education_self",
    "Q276": "education_spouse",
    "Q277": "education_mother",
    "Q278": "education_father",
    "Q279": "employment_self",
    "Q280": "employment_spouse",
    "Q281": "occupation_self",
    "Q282": "occupation_spouse",
    "Q283": "occupation_father14",
    "Q284": "work_sector",
    "Q285": "chief_wage_earner",
    "Q286": "saving_behavior",
    "Q287": "social_class",
    "Q288": "income_decile",
    "Q289": "religion",
    "Q290": "ethnic_group",
}

df = pd.read_csv(input_csv, dtype=str)

available_cols = [col for col in column_map.keys() if col in df.columns]
df_subset = df[available_cols].rename(columns=column_map)


def try_int(val):
    try:
        if pd.isna(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return val


df_subset = df_subset.map(try_int)

records = df_subset.where(df_subset.notnull(), None).to_dict(orient="records")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved JSON file: {output_json}  (total {len(records)} records)")
