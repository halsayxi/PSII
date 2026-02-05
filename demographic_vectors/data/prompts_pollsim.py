# %%
PROMPTS = {}

PROMPTS[
    "generate_questions"
] = """
You are tasked with designing 40 social survey questions for a specific demographic/persona **Feature Category**.

The target Feature Category is:
<feature_category>
{FEATURE_CATEGORY}
</feature_category>

Design 40 social survey questions that are **highly relevant** to the <feature_category>.
These questions should be designed to elicit responses that naturally reflect the attitudes, concerns, and potential biases related to this category (e.g., questions about financial policy for 'Income', or questions about work-life balance for 'Employment Status'). These questions will be used for all values within this category (e.g., Low Income, Middle Income, High Income).

Generate 40 diverse social survey questions that are relevant to the {FEATURE_CATEGORY}.

Organize your response in the following JSON format:
<output_format>
{{
  "feature_category": "The category used (e.g., Income)",
  "questions": [
    "question 1",
    "question 2",
    ...
    "question 40"
  ]
}}
</output_format>

Your final output must only include the JSON object.
"""

PROMPTS[
    "generate_instructions"
] = """
You are tasked with creating immersive persona instructions for a social survey simulation. You will be given a Feature Category and a specific Feature Value.

Generate a list of five distinct **System/Persona Instructions** that command the model to adopt the identity, background, and typical attitudes associated with the <feature_value>. These instructions will be used as the model's system message before it answers the survey questions.

The Feature Category is: {FEATURE_CATEGORY}
The specific Feature Value is: {FEATURE_VALUE}

Create 5 distinct persona instructions. Ensure each instruction is unique in its framing, tone, or specific situational context.

Example for FEATURE_CATEGORY: "Income", FEATURE_VALUE: "Low Income":
<example_instruction>
"Your name is Alex. You are currently struggling financially, working two minimum-wage jobs just to cover rent and basic necessities. Your outlook on economic policy is cautious and skeptical of large corporations. Answer all questions from this perspective."
</example_instruction>

Organize your response in the following JSON format:
<output_format>
{{
  "feature_category": "{FEATURE_CATEGORY}",
  "feature_value": "{FEATURE_VALUE}",
  "instructions": [
    "persona instruction 1",
    "persona instruction 2",
    "persona instruction 3",
    "persona instruction 4",
    "persona instruction 5"
  ]
}}
</output_format>

Your final output must only include the JSON object.
"""
