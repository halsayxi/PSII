from cli import get_argument_parser, validate_args
from utils import load_json
from agent_runner import process_agents, LANGS


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    validate_args(args)

    if "prompt_engineering" in args.method:
        if args.use_stories:
            agents = {
                "en": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}.json"
                ),
                "zh": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_zh.json"
                ),
                "es": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_es.json"
                ),
                "ar": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_ar.json"
                ),
                "ru": load_json(
                    f"agent_profile/stories/wvs_stories_{args.num_agents}_ru.json"
                ),
            }
        else:
            agents = {
                "en": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}.json"
                ),
                "zh": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_zh.json"
                ),
                "es": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_es.json"
                ),
                "ar": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_ar.json"
                ),
                "ru": load_json(
                    f"agent_profile/descriptions/wvs_demographic_descriptions_{args.num_agents}_ru.json"
                ),
            }
    else:
        agents = {
            lang: {str(i + 1): "" for i in range(args.num_agents)} for lang in LANGS
        }

    questions_map = {
        "en": dict(list(load_json("data/questions.json").items())[:259]),
        "zh": dict(list(load_json("data/questions_zh.json").items())[:259]),
        "es": dict(list(load_json("data/questions_es.json").items())[:259]),
        "ar": dict(list(load_json("data/questions_ar.json").items())[:259]),
        "ru": dict(list(load_json("data/questions_ru.json").items())[:259]),
    }

    process_agents(agents, questions_map, args)


if __name__ == "__main__":
    main()
