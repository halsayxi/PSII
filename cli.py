import argparse

VECTOR_KEYWORD = "vector"
DEMOGRAPHIC_KEYWORD = "demographic_vectors"
VALUE_KEYWORD = "value_vector"


def parse_vector_q_codes(s):
    """
    Format:
      qcode:layer,qcode:layer,...
      Example: 1:20,2:18,3:14

    Returns:
      dict[qcode] = layer (int)
    """
    result = {}
    for item in s.split(","):
        if ":" not in item:
            raise ValueError(f"Each qcode must have a layer: {item}")
        q, l = item.split(":")
        result[int(q)] = int(l)
    return result


CHOICES = [
    "direct",
    "prompt_engineering",
    "multilingual",
    "requesting_diversity",
    "demographic_vectors",
    "value_vector",
]


def parse_methods(s):
    methods = [m.strip() for m in s.split(",") if m.strip()]
    invalid = [m for m in methods if m not in CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid method(s): {invalid}. Must be one of {CHOICES}"
        )
    sorted_methods = [m for m in CHOICES if m in methods]
    return "_".join(sorted_methods)


def get_argument_parser():
    parser = argparse.ArgumentParser(description="PollSim Arguments")
    parser.add_argument(
        "--model_name", type=str, default="qwen2.5-7b", help="Name of the model to use."
    )
    parser.add_argument(
        "--use_local_model",
        action="store_true",
        help="Whether to use a local model checkpoint instead of API.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum number of tokens to generate per response (for local model).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for model inference.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum number of retry attempts if generation fails.",
    )
    parser.add_argument(
        "--num_agents", type=int, default=100, help="Number of simulated agents."
    )
    parser.add_argument(
        "--coef", type=float, help="Coefficient for steering vector scaling."
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        help="Standard deviation of Gaussian noise added to vectors.",
    )
    parser.add_argument(
        "--steering_type",
        type=str,
        choices=["response", "prompt", "all"],
        help="Type of steering to apply: response, prompt, or all.",
    )
    parser.add_argument(
        "--demographic_vectors_dir",
        type=str,
        default="demographic_vectors/vectors_qwen2.5-7b",
        help="Directory containing demographic vectors.",
    )
    parser.add_argument(
        "--value_vectors_dir",
        type=str,
        default="value_vectors/vectors",
        help="Directory containing value vectors.",
    )
    parser.add_argument(
        "--vector_q_codes",
        type=parse_vector_q_codes,
        help="Vector qcodes with layer, format: qcode:layer,qcode:layer,...",
    )
    parser.add_argument(
        "--use_stories",
        action="store_true",
        help="Whether to use story-based context for agents.",
    )
    parser.add_argument(
        "--method",
        type=parse_methods,
        default="prompt_engineering",
        help="Method(s) to use, comma-separated. Multiple methods will be sorted and joined by underscores.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs."
    )
    parser.add_argument(
        "--save_hidden_states",
        action="store_true",
        help="Whether to save hidden states from the model.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Whether to process agents in reverse order.",
    )
    return parser


def is_vector_method(method: str) -> bool:
    return VECTOR_KEYWORD in method


def is_demographic_method(method: str) -> bool:
    return DEMOGRAPHIC_KEYWORD in method


def is_value_vector_method(method: str) -> bool:
    return VALUE_KEYWORD in method


def validate_args(args):
    if not args.use_local_model and args.max_new_tokens is not None:
        raise ValueError("--max_new_tokens requires --use_local_model")
    if args.use_stories and "prompt_engineering" not in args.method:
        raise ValueError(
            "--use_stories can only be used with prompt-engineering methods"
        )

    demographic_vector_required = ["coef", "steering_type", "vector_q_codes"]
    non_demographic_vector_forbidden = [
        "coef",
        "steering_type",
        "noise_std",
        "vector_q_codes",
    ]

    if is_demographic_method(args.method):
        for name in demographic_vector_required:
            if getattr(args, name) is None:
                raise ValueError(f"--{name} is required when method='{args.method}'")
    else:
        for name in non_demographic_vector_forbidden:
            if getattr(args, name) is not None:
                raise ValueError(
                    f"--{name} is only allowed for demographic_vector-based methods"
                )
