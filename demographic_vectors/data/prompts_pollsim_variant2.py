# more simple / direct
PROMPTS = {}

PROMPTS["generate_questions"] = """
Create 40 social survey questions for a demographic/persona **Feature Category**.

Category:
{FEATURE_CATEGORY}

The questions should:
- Be clearly related to the category
- Capture attitudes, concerns, and potential biases
- Apply to all values within this category

Ensure diversity across the 40 questions.

Return JSON:
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "questions": ["...", "...", "..."]
}}

Only output JSON.
"""

PROMPTS["generate_instructions"] = """
Generate 5 persona instructions for a survey simulation.

Category: {FEATURE_CATEGORY}
Value: {FEATURE_VALUE}

Each instruction should:
- Define identity and background
- Reflect realistic attitudes
- Be distinct from others

Return JSON:
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "feature_value": "{FEATURE_VALUE}",
  "instructions": ["...", "...", "...", "...", "..."]
}}

Only output JSON.
"""