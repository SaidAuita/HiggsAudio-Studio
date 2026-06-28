"""AI-режиссёр текста: нормализация+теги (роль A), сценарий подкаста (B), кастинг аудиокниги (C).



LLM — Qwen3.5 GGUF (4-бит Q4_K_M) через llama.cpp на GPU (n_gpu_layers=-1).

CUDA-DLL кладутся install.bat рядом с llama.dll, поэтому импорт без трюков.

Фильтр тегов и тесты работают без llama.cpp/GPU (ленивые импорты).

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

_CPU_MODE = False

def set_cpu_mode(val):
    global _CPU_MODE
    _CPU_MODE = bool(val)





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





MODELS = {  # GGUF (llama.cpp, GPU): качается 4-бит Q4_K_M, НЕ полный bf16

    "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),

    "Qwen3.5-4B · Q4_K_M (лёгкая, ~2.5 ГБ)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),

}

DEFAULT_MODEL = "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)"





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



_daemon_proc = None



def get_daemon():

    global _daemon_proc

    if _daemon_proc is not None and _daemon_proc.poll() is None:

        return _daemon_proc

        

    import subprocess

    import sys

    py_exe = sys.executable

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "director_daemon.py")

    

    print(f"[director] starting daemon process...", flush=True)

    _daemon_proc = subprocess.Popen(

        [py_exe, "-u", script_path],

        stdin=subprocess.PIPE,

        stdout=subprocess.PIPE,

        stderr=subprocess.DEVNULL,

        text=True,

        encoding="utf-8",

        bufsize=1

    )

    return _daemon_proc



def load_llm(label=DEFAULT_MODEL):

    """(Deprecated) Daemon manages model loading now."""

    pass



def is_keep_vram():

    try:

        import json

        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_config.json")

        if os.path.exists(cfg_path):

            with open(cfg_path, "r", encoding="utf-8") as f:

                return json.load(f).get("keep_vram", False)

    except Exception:

        pass

    return False





def unload_llm(force=False):

    if is_keep_vram() and not force:

        return

    global _daemon_proc

    if _daemon_proc is not None and _daemon_proc.poll() is None:

        print("[director] unloading LLM daemon process...", flush=True)

        try:

            import json

            _daemon_proc.stdin.write(json.dumps({"action": "unload"}) + "\n")

            _daemon_proc.stdin.flush()

            _daemon_proc.stdout.readline()

            _daemon_proc.wait(timeout=5)

        except Exception:

            pass

        if _daemon_proc is not None and _daemon_proc.poll() is None:

            try:

                _daemon_proc.terminate()

                _daemon_proc.wait(timeout=2)

            except Exception:

                pass

    _daemon_proc = None





CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")



def load_llm_config():

    defaults = {

        "api_url": "http://127.0.0.1:1234/v1",

        "api_key": "",

        "api_model": "gemma-2-12b-it",

        "system_prompt": (

            "Ты — режиссёр озвучки. Нормализуй текст под произношение (числа, даты, аббревиатуры, "

            "валюты, единицы, символы — словами), исправь явные опечатки, и расставь эмоциональные / "

            "sfx / prosody-теги по смыслу. ОБЯЗАТЕЛЬНО сохраняй исходный язык текста: если текст "

            "на английском — возвращай текст на английском, если на русском — на русском, не переводи его."

        )

    }

    if os.path.exists(CONFIG_PATH):

        try:

            import json

            with open(CONFIG_PATH, "r", encoding="utf-8") as f:

                data = json.load(f)

                defaults.update(data)

        except Exception as e:

            print(f"[director] Error loading llm_config: {e}")

    return defaults





def _strip_think(txt):

    return re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()





def _chat(system, user, max_new=1024, temp=0.4, label=DEFAULT_MODEL, api_url=None, api_key=None, api_model=None):

    if _MOCK:

        return user  # в mock — проброс без изменений



    if label == "External API (LM Studio / Ollama / OpenAI)":

        import urllib.request

        import urllib.error

        import json

        

        cfg = load_llm_config()

        active_url = api_url or cfg["api_url"] or "http://127.0.0.1:1234/v1"

        active_key = api_key or cfg["api_key"]

        active_model = api_model or cfg["api_model"]



        url = active_url.rstrip("/") + "/chat/completions"

        headers = {

            "Content-Type": "application/json"

        }

        if active_key:

            headers["Authorization"] = f"Bearer {active_key}"

        data = {

            "messages": [

                {"role": "system", "content": system},

                {"role": "user", "content": user}

            ],

            "temperature": temp,

            "max_tokens": max_new,

            "stream": False

        }

        if active_model:

            data["model"] = active_model

        

        req_data = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

        

        print(f"[director] Sending request to external API: {url} (model: {active_model})", flush=True)

        try:

            with urllib.request.urlopen(req, timeout=30) as response:

                res_data = response.read().decode("utf-8")

                res = json.loads(res_data)

                content = res["choices"][0]["message"]["content"]

                print(f"[director] Response received successfully. Content length: {len(content)}", flush=True)

                return _strip_think(content)

        except urllib.error.HTTPError as e:

            try:

                err_body = e.read().decode("utf-8")

                detail = json.loads(err_body)

                err_msg = detail.get("error", {}).get("message", str(e))

            except Exception:

                err_msg = str(e)

            print(f"[director] External API HTTP error {e.code}: {err_msg}", flush=True)

            raise RuntimeError(f"Ошибка внешнего API LLM (HTTP {e.code}): {err_msg}")

        except urllib.error.URLError as e:

            print(f"[director] External API connection error: {e.reason}", flush=True)

            raise RuntimeError(f"Ошибка подключения к API: {e.reason}. Убедитесь, что LM Studio / Ollama запущен.")

        except Exception as e:

            print(f"[director] External API error: {e}", flush=True)

            raise RuntimeError(f"Ошибка внешнего API LLM: {e}")



    import json

    proc = get_daemon()

    req = {

        "action": "chat",

        "system": system,

        "user": user,

        "max_new": max_new,

        "temp": temp,

        "label": label,

        "use_cpu": _CPU_MODE

    }

    

    try:

        proc.stdin.write(json.dumps(req) + "\n")

        proc.stdin.flush()

        line = proc.stdout.readline()

    except Exception as e:

        print(f"[director] daemon error: {e}. Attempting restart...", flush=True)

        unload_llm()

        proc = get_daemon()

        proc.stdin.write(json.dumps(req) + "\n")

        proc.stdin.flush()

        line = proc.stdout.readline()

        

    if not line:

        raise RuntimeError("Director daemon exited unexpectedly")

        

    res = json.loads(line)

    if res.get("status") == "success":

        return res["content"]

    else:

        raise RuntimeError(f"Director daemon error: {res.get('error')}")





def _filter_line(line):

    """Сохранить префикс 'ИМЯ:', отфильтровать теги в произносимой части."""

    if ":" in line:

        who, _, said = line.partition(":")

        return f"{who}:{filter_tags(said)}"

    return filter_tags(line)





def enrich(text, label=DEFAULT_MODEL):

    '''РОЛЬ A — нормализация под произношение + лёгкая правка + теги по смыслу.'''

    cfg = load_llm_config()

    s = cfg["system_prompt"] + "\n" + _TAG_RULES

    return filter_tags(_chat(s, text, label=label))





def write_podcast(topic, n_speakers=2, label=DEFAULT_MODEL):

    """РОЛЬ B — мульти-спикерный диалог в индексном формате 'Speaker N: реплика' (как Qwen3-TTS)."""

    n = max(2, int(n_speakers))

    s = (f"Ты — сценарист подкаста на {n} спикеров (Speaker 0 .. Speaker {n - 1}). "

         "Напиши живой диалог: КАЖДАЯ строка строго в формате 'Speaker K: реплика', где K — номер от 0. "

         "Дай каждому спикеру свою манеру речи и характер. В репликах расставляй теги по смыслу. "

         "Пиши на том же языке, на котором задана тема (если на английском — пиши на английском, "

         "если на русском — на русском). " + _TAG_RULES)

    out = _chat(s, topic, max_new=2048, label=label)

    return "\n".join(_filter_line(ln) for ln in out.splitlines())





def cast_audiobook(text, n_voices=2, label=DEFAULT_MODEL):

    """РОЛЬ C — атрибуция в индексном формате: Speaker 0 = рассказчик, 1.. = персонажи."""

    n = max(2, int(n_voices))

    s = ("Ты — кастинг-режиссёр аудиокниги. Раздели текст на речь рассказчика и реплики персонажей. "

         f"Speaker 0 — РАССКАЗЧИК (авторский текст), Speaker 1 .. Speaker {n - 1} — персонажи "

         "(закрепи за каждым персонажем свой номер и держи его постоянным). "

         "КАЖДАЯ строка строго в формате 'Speaker K: реплика'. Текст сохраняй ДОСЛОВНО на его исходном языке, "

         "не переводи и не изменяй слова, только размечай говорящего и добавляй теги по смыслу. " + _TAG_RULES)

    out = _chat(s, text, max_new=2048, label=label)

    return "\n".join(_filter_line(ln) for ln in out.splitlines())

