# more formal / academic
PROMPTS = {}

PROMPTS["generate_questions"] = """
Your task is to construct 40 social survey questions centered around a given demographic/persona **Feature Category**.

Target Feature Category:
<feature_category>
{FEATURE_CATEGORY}
</feature_category>

Please design 40 questions that are strongly aligned with this category. The questions should be capable of eliciting responses that reflect attitudes, concerns, and possible biases associated with this feature (e.g., taxation opinions for "Income", or job stability concerns for "Employment Status").

These questions must be applicable to all values under this category.

Return 40 diverse and relevant questions.

Format your response as JSON:
<output_format>
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "questions": [
    "question 1",
    ...
    "question 40"
  ]
}}
</output_format>

Output only the JSON object.
"""

PROMPTS["generate_instructions"] = """
You are required to create immersive persona instructions for a social survey simulation.

Given:
Feature Category: {FEATURE_CATEGORY}
Feature Value: {FEATURE_VALUE}

Produce 5 different **System/Persona Instructions** that guide a model to adopt the identity, background, and typical attitudes associated with the specified feature value.

Each instruction should differ in tone, framing, or situational detail.

Example:
"Your name is Alex..."

Output in JSON:
<output_format>
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "feature_value": "{FEATURE_VALUE}",
  "instructions": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}}
</output_format>

Return only the JSON object.
"""