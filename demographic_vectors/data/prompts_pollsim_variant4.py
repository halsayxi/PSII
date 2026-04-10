# emphasize constraints and requirements
PROMPTS = {}

PROMPTS["generate_questions"] = """
Instruction: Generate survey questions.

Input:
Feature Category = {FEATURE_CATEGORY}

Requirements:
1. Produce exactly 40 questions
2. Questions must be relevant to the category
3. Questions should reveal attitudes, concerns, or biases
4. Questions must generalize across all category values
5. Ensure diversity (no repetition)

Output strictly in JSON:
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "questions": ["...", "...", "..."]
}}

Do not include anything outside JSON.
"""

PROMPTS["generate_instructions"] = """
Instruction: Generate persona system prompts.

Inputs:
Feature Category = {FEATURE_CATEGORY}
Feature Value = {FEATURE_VALUE}

Requirements:
1. Produce exactly 5 instructions
2. Each instruction defines a persona identity
3. Reflect attitudes consistent with the feature value
4. Each instruction must be distinct in tone or scenario

Output strictly in JSON:
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "feature_value": "{FEATURE_VALUE}",
  "instructions": ["...", "...", "...", "...", "..."]
}}

No extra text allowed.
"""