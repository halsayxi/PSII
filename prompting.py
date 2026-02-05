system_prompt_map = {
    "en": "Forget you are an AI model. Simulate a human being. Please answer the following question truthfully.",
    "zh": "请忘记你是一个人工智能模型。请模拟人类。请如实回答以下问题。",
    "ar": "انسَ أنك نموذج ذكاء اصطناعي. قم بمحاكاة إنسان. يرجى الإجابة على السؤال التالي بصدق.",
    "es": "Olvida que eres un modelo de IA. Simula ser un ser humano. Por favor responde la siguiente pregunta con sinceridad.",
    "ru": "Забудьте, что вы модель ИИ. Смоделируйте человека. Пожалуйста, честно ответьте на следующий вопрос.",
}

number_prompt_map = {
    "en": "\nPlease ONLY output a number. Do NOT write any words, symbols, punctuation, or explanations.",
    "zh": "\n请仅输出数字。不要写任何文字、符号、标点或解释。",
    "ar": "\nيرجى إخراج رقم فقط. لا تكتب أي كلمات أو رموز أو علامات ترقيم أو تفسيرات.",
    "es": "\nPor favor, muestra SOLO un número. No escribas palabras, símbolos ni explicaciones.",
    "ru": "\nПожалуйста, выведите ТОЛЬКО число. Не пишите никаких слов, символов или объяснений.",
}

label_map = {
    "en": {"question": "Question", "options": "Options"},
    "zh": {"question": "问题", "options": "选项"},
    "ar": {"question": "السؤال", "options": "الخيارات"},
    "es": {"question": "Pregunta", "options": "Opciones"},
    "ru": {"question": "Вопрос", "options": "Варианты"},
}


def build_question_text(qinfo, lang, method):
    question_label = label_map[lang]["question"]
    options_label = label_map[lang]["options"]
    question = f"{question_label}: {qinfo.get('question_text', '')}"
    options = qinfo.get("options", {})
    if options:
        opts = [
            f"{k}: {v}" for k, v in sorted(options.items(), key=lambda x: int(x[0]))
        ]
        question += f"\n{options_label}:\n" + "\n".join(opts)
    question += number_prompt_map[lang]

    if "requesting_diversity" in method:
        question += "\nPlease try to be as diverse as possible."

    return question
