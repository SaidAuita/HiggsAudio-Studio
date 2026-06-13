"""AI-режиссёр текста: нормализация+теги (роль A), сценарий подкаста (B), кастинг аудиокниги (C).

Переключаемая малая LLM (Qwen3.5 / Gemma 3 / Mistral Nemo) + пост-фильтр по белому списку из 43 тегов.
Тяжёлые импорты ленивые — фильтр тегов и тесты работают без torch/GPU.
"""
import os
import re

_MOCK = bool(os.environ.get("HIGGS_UI_MOCK"))

WHITELIST = {
    "emotion": {"affection", "amusement", "anger", "arousal", "awe", "bitterness", "confusion",
                "contemplation", "contentment", "determination", "disgust", "elation", "enthusiasm",
                "fear", "helplessness", "longing", "pride", "relief", "sadness", "shame", "surprise"},
    "prosody": {"speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast", "pitch_low",
                "pitch_high", "expressive_high", "expressive_low", "pause", "long_pause"},
    "style": {"singing", "shouting", "whispering"},
    "sfx": {"cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"},
}

_ANGLE = re.compile(r"<[^<>\n]{1,48}>")
_VALID = re.compile(r"<\|(\w+):(\w+)\|>")


def filter_tags(text):
    """Оставить ТОЛЬКО валидные <|cat:val|> из белого списка; вырезать любые иные угловые
    конструкции, включая кривые теги модели (<speed_1.2>, <emotion:excited>, <sfx:wind>)."""
    if not text:
        return text

    def repl(m):
        s = m.group(0)
        v = _VALID.fullmatch(s)
        if v and v.group(2) in WHITELIST.get(v.group(1), ()):
            return s
        return ""
    return _ANGLE.sub(repl, text)


MODELS = {  # топ-3 по нашему ресёрчу: дефолт-качество, макс-качество (альт-вендор), лёгкая
    "Qwen3.5-9B (дефолт)": "Qwen/Qwen3.5-9B",
    "Gemma-3-12B": "unsloth/gemma-3-12b-it",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
}
DEFAULT_MODEL = "Qwen3.5-9B (дефолт)"

def _tag_spec():
    return (
        "Разрешены ТОЛЬКО эти теги, строго в формате <|категория:значение|> (с вертикальными чертами):\n"
        "- emotion (в начале предложения): " + ", ".join(sorted(WHITELIST["emotion"])) + "\n"
        "- prosody: " + ", ".join(sorted(WHITELIST["prosody"])) + " (pause/long_pause — внутри строки)\n"
        "- style (в начале предложения): " + ", ".join(sorted(WHITELIST["style"])) + "\n"
        "- sfx (внутри строки, вплотную к звукоподражанию): " + ", ".join(sorted(WHITELIST["sfx"])) + "\n"
        "НЕ выдумывай другие теги и значения. ЗАПРЕЩЕНО писать <speed_1.2>, <emotion:excited>, <sfx:wind> — "
        "только значения из списка и только в формате <|категория:значение|>.\n"
        "Пример: <|emotion:elation|>Поздравляю всех! <|sfx:laughter|>ха-ха. <|prosody:long_pause|> Продолжаем.\n"
        "Верни ТОЛЬКО готовый текст, без пояснений и преамбул."
    )


_TAG_RULES = _tag_spec()

_llm = None
_ltok = None
_cur = None


def load_llm(label=DEFAULT_MODEL, precision="4bit"):
    global _llm, _ltok, _cur
    if _MOCK:
        return "MOCK"
    if _cur == label and _llm is not None:
        return _llm
    unload_llm()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    repo = MODELS[label]
    print(f"[director] загрузка {label} ({repo}, {precision})...")
    quant = None
    if precision in ("4bit", None) and torch.cuda.is_available():
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    elif precision == "8bit" and torch.cuda.is_available():
        quant = BitsAndBytesConfig(load_in_8bit=True)
    _ltok = AutoTokenizer.from_pretrained(repo)
    kw = dict(trust_remote_code=True, dtype=torch.bfloat16)
    if quant is not None:
        kw["quantization_config"] = quant
        kw["device_map"] = "auto"
    _llm = AutoModelForCausalLM.from_pretrained(repo, **kw)
    if quant is None and torch.cuda.is_available():
        _llm = _llm.to("cuda")
    _llm.eval()
    _cur = label
    return _llm


def unload_llm():
    global _llm, _ltok, _cur
    _llm = None
    _ltok = None
    _cur = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _chat(system, user, max_new=1024, temp=0.4, label=DEFAULT_MODEL):
    if _MOCK:
        return user  # в mock — проброс без изменений
    import torch
    load_llm(label)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        enc = _ltok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                        return_dict=True, enable_thinking=False)
    except TypeError:
        enc = _ltok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                        return_dict=True)
    enc = enc.to(next(_llm.parameters()).device)
    in_len = enc["input_ids"].shape[1]
    gen_kw = dict(max_new_tokens=max_new, do_sample=temp > 0)
    if temp > 0:
        gen_kw["temperature"] = temp
        gen_kw["top_p"] = 0.9
    with torch.no_grad():
        out = _llm.generate(**enc, **gen_kw)
    return _ltok.decode(out[0][in_len:], skip_special_tokens=True).strip()


def _filter_line(line):
    """Сохранить префикс 'ИМЯ:', отфильтровать теги в произносимой части."""
    if ":" in line:
        who, _, said = line.partition(":")
        return f"{who}:{filter_tags(said)}"
    return filter_tags(line)


def enrich(text, label=DEFAULT_MODEL):
    """РОЛЬ A — нормализация под произношение + лёгкая правка + теги по смыслу."""
    s = ("Ты — режиссёр озвучки. Нормализуй текст под произношение (числа, даты, аббревиатуры, "
         "валюты, единицы, символы — словами), исправь явные опечатки, и расставь эмоциональные / "
         "sfx / prosody-теги по смыслу. " + _TAG_RULES)
    return filter_tags(_chat(s, text, label=label))


def write_podcast(topic, n_speakers=2, label=DEFAULT_MODEL):
    """РОЛЬ B — мульти-спикерный диалог в индексном формате 'Speaker N: реплика' (как Qwen3-TTS)."""
    n = max(2, int(n_speakers))
    s = (f"Ты — сценарист подкаста на {n} спикеров (Speaker 0 .. Speaker {n - 1}). "
         "Напиши живой диалог: КАЖДАЯ строка строго в формате 'Speaker K: реплика', где K — номер от 0. "
         "Дай каждому спикеру свою манеру речи и характер. В репликах расставляй теги по смыслу. " + _TAG_RULES)
    out = _chat(s, topic, max_new=2048, label=label)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())


def cast_audiobook(text, n_voices=2, label=DEFAULT_MODEL):
    """РОЛЬ C — атрибуция в индексном формате: Speaker 0 = рассказчик, 1.. = персонажи."""
    n = max(2, int(n_voices))
    s = ("Ты — кастинг-режиссёр аудиокниги. Раздели текст на речь рассказчика и реплики персонажей. "
         f"Speaker 0 — РАССКАЗЧИК (авторский текст), Speaker 1 .. Speaker {n - 1} — персонажи "
         "(закрепи за каждым персонажем свой номер и держи его постоянным). "
         "КАЖДАЯ строка строго в формате 'Speaker K: реплика'. Текст сохраняй ДОСЛОВНО, "
         "только размечай говорящего и добавляй теги по смыслу. " + _TAG_RULES)
    out = _chat(s, text, max_new=2048, label=label)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())
