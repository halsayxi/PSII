# emphasize reasoning and intent
PROMPTS = {}

PROMPTS["generate_questions"] = """
You are designing survey instruments for social simulation.

Feature Category:
{FEATURE_CATEGORY}

Generate 40 questions that probe beliefs, preferences, and behavioral tendencies related to this category.

The questions should implicitly capture:
- underlying attitudes
- real-world concerns
- potential biases

They must remain applicable across all possible values of this category.

Output format:
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "questions": ["...", "...", "..."]
}}

Return only the JSON.
"""

PROMPTS["generate_instructions"] = """
You are simulating human personas in a survey environment.

Given:
Feature Category: {FEATURE_CATEGORY}
Feature Value: {FEATURE_VALUE}

Write 5 system instructions that make the model behave like a person with this attribute.

Each instruction should:
- embed identity
- reflect lifestyle or context
- influence responses naturally

Ensure variation in style and framing.

Output:
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "feature_value": "{FEATURE_VALUE}",
  "instructions": ["...", "...", "...", "...", "..."]
}}

Only return JSON.
"""