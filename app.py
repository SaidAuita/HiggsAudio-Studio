"""Higgs Audio Studio — портативная сборка Nerual Dreming + Нейро-Софт.

Higgs Audio v3.1 TTS + FireRedTTS3 TTS (100+ языков, чистое клонирование, Voice Design, Speech Edit)
+ AI-режиссёр текста (теги по смыслу) + мульти-спикерные режимы Подкаст и Аудиокнига. UI RU/EN, тёмная тема.
"""
import os
import re
import warnings

# Suppress deprecation and user warnings in third-party libraries (Gradio/Starlette/Torch)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import asyncio

# Корень проекта в sys.path — embedded python не добавляет каталог скрипта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Windows retry_open патч для anyio/aiofiles (PermissionError от антивируса)
if sys.platform == "win32":
    try:
        import anyio
        import anyio._core._fileio
        _orig_open = anyio._core._fileio.open_file

        async def _retry_open(file, *a, **k):
            delay = 0.2
            for i in range(20):
                try:
                    return await _orig_open(file, *a, **k)
                except PermissionError:
                    if i == 19:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 1.2

        anyio._core._fileio.open_file = _retry_open
        anyio.open_file = _retry_open
    except Exception:
        pass

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

from datetime import datetime
from pathlib import Path

import gradio as gr

import higgs_engine as eng
import firered_engine as fe
import director as dr

SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "output"
VOICES_DIR = SCRIPT_DIR / "voices"
OUTPUT_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)

CONFIG_PATH = SCRIPT_DIR / "llm_config.json"


def load_llm_config():
    defaults = {
        "api_url": "http://localhost:1234/v1",
        "api_key": "",
        "api_model": "gemma-2-12b-it",
        "system_prompt": (
            "Ты — режиссёр озвучки. Нормализуй текст под произношение (числа, даты, аббревиатуры, "
            "валюты, единицы, символы — словами), исправь явные опечатки, и расставь эмоциональные / "
            "sfx / prosody-теги по смыслу. ОБЯЗАТЕЛЬНО сохраняй исходный язык текста: если текст "
            "на английском — возвращай текст на английском, если на русском — на русском, не переводи его."
        )
    }
    if CONFIG_PATH.exists():
        try:
            import json
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                defaults.update(data)
        except Exception as e:
            print(f"Error loading llm_config: {e}")
    return defaults


def save_llm_config(api_url, api_key, api_model, system_prompt):
    try:
        import json
        data = {
            "api_url": api_url,
            "api_key": api_key,
            "api_model": api_model,
            "system_prompt": system_prompt
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return "Настройки LLM сохранены / LLM settings saved"
    except Exception as e:
        return f"Ошибка сохранения / Save error: {e}"


APP_NAME = "Higgs Audio Studio & FireRedTTS3"
DEVICE_INFO = eng.device_info()
MODEL_CHOICES = list(dr.MODELS.keys()) + ["External API (LM Studio / Ollama / OpenAI)"]

ENGINE_HIGGS = "🎙️ Higgs Audio v3.1 (Эмоции & SFX / 4B)"
ENGINE_FIRERED = "🔥 FireRedTTS3 (Чистый звук / DiT Flow / 24 языка)"
ENGINE_CHOICES = [ENGINE_HIGGS, ENGINE_FIRERED]

MAX_SPK = 4
OWN_FILE = "— свой файл / own file —"

CLOUD_VOICES_REPO = "Slait/russia_voices"
CLOUD_VOICES_BASE = "https://huggingface.co/datasets/Slait/russia_voices/resolve/main"


# ----------------------------------------------------------------------------
# Брендинг
# ----------------------------------------------------------------------------
_FLAG = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg"


def _brand(subtitle):
    return f"""
<div class="brand-header">
  <div class="lang-switcher">
    <a href="?__lang=ru&amp;__theme=dark" onclick="window.__setLang('ru'); return false;" class="lang-btn"><img src="{_FLAG}/1f1f7-1f1fa.svg" width="16" height="16"/>RU</a>
    <a href="?__lang=en&amp;__theme=dark" onclick="window.__setLang('en'); return false;" class="lang-btn"><img src="{_FLAG}/1f1ec-1f1e7.svg" width="16" height="16"/>EN</a>
  </div>
  <div class="brand-box">
    <div class="brand-title">🎙️ {APP_NAME}</div>
    <div class="brand-subtitle">{subtitle}</div>
    <div class="device-badge">💻 {DEVICE_INFO}</div>
  </div>
</div>
"""


BRAND_HTML_RU = _brand(
    "Higgs Audio v3.1 (100+ языков) + FireRedTTS3 (24+ языков) · Клонирование · Voice Design · Speech Edit · AI-режиссёр"
)
BRAND_HTML_EN = _brand(
    "Higgs Audio v3.1 (100+ languages) + FireRedTTS3 (24+ languages) · Voice Cloning · Voice Design · Speech Edit · AI Director"
)


def _legend(emo, pro, sty, sfx, fmt):
    w = dr.WHITELIST
    return (f"**{emo}** " + ", ".join(sorted(w["emotion"])) + "\n\n"
            f"**{pro}** " + ", ".join(sorted(w["prosody"])) + "\n\n"
            f"**{sty}** " + ", ".join(sorted(w["style"])) + "\n\n"
            f"**{sfx}** " + ", ".join(sorted(w["sfx"])) + "\n\n" + fmt)


_LEGEND_RU = _legend("Эмоции (в начале предложения):", "Просодия:", "Стиль:",
                     "Звуки (внутри строки, тег вплотную):",
                     "Формат: `<|category:value|>`. Пример: `<|emotion:elation|>Привет! <|sfx:laughter|>ха-ха`")
_LEGEND_EN = _legend("Emotions (at sentence start):", "Prosody:", "Style:", "Sounds (inline, tag attached):",
                     "Format: `<|category:value|>`. Example: `<|emotion:elation|>Hi! <|sfx:laughter|>ha-ha`")

_PODFMT_RU = "**Формат сценария:** каждая строка `Speaker 0: реплика` / `Speaker 1: …` (также `Диктор N:`, `[N]`). Номер = диктор ниже."
_PODFMT_EN = "**Script format:** one line per turn `Speaker 0: line` / `Speaker 1: …` (also `[N]`). The number = speaker below."
_BOOKFMT_RU = "**Формат:** `Speaker 0:` — рассказчик, `Speaker 1+:` — персонажи. Разметь кнопкой или вручную, потом озвучь."
_BOOKFMT_EN = "**Format:** `Speaker 0:` — narrator, `Speaker 1+:` — characters. Attribute with the button or by hand, then synthesize."


# ----------------------------------------------------------------------------
# i18n (gr.I18n)
# ----------------------------------------------------------------------------
_RU = {
    "engine_higgs_tab": "🎙️ Higgs Audio v3.1 (100+ языков, Эмоции & Режиссёр)",
    "engine_firered_tab": "🔥 FireRedTTS3 (24 языка, Voice Design, Редактирование речи)",
    "tab_tts": "🎙️ Озвучка", "tab_expr": "🎭 Экспрессия + Режиссёр", "tab_clone": "🧬 Клонирование",
    "tab_design": "🎨 Создание голоса", "tab_edit": "✂️ Редактирование речи",
    "tab_pod": "🎬 Подкаст", "tab_book": "📚 Аудиокнига", "tab_batch": "📦 Пакет",
    "engine_label": "Движок озвучки (TTS Engine)",
    "lang_label": "Язык (для FireRedTTS3)",
    "text": "Текст", "ph_text": "Введите текст…",
    "generate": "🔊 Озвучить", "stop": "⏹ Стоп", "result": "Результат", "advanced": "Доп. настройки",
    "director_model": "Модель режиссёра (для обогащения / диалогов)",
    "quant": "Квантизация / Точность",
    "quant_info": "bf16 — наилучшее качество. 8-bit / int8 — экономит VRAM при сохранении чистоты. 4-bit — для карт с малым объёмом памяти.",
    "out_format": "Формат вывода",
    "cat_emotion": "😊 Эмоции (на предложение)", "cat_prosody": "🎵 Просодия", "cat_style": "🎭 Стиль", "cat_sfx": "🔊 Звуки (по месту в тексте)",
    "download_all": "⬇️ Скачать все 700+",
    "enrich": "✨ Обогатить текст", "auto_enrich": "✨ Авто-обогащение промпта режиссёром",
    "upload_txt": "📂 Загрузить текст из файла (.txt)",
    "ref_voice": "Аудио-референс (голос)", "ref_text": "Транскрипт референса (заполнится сам)",
    "ph_clone_tr": "Что произносится в референсе…", "voice_preset": "Пресет голоса",
    "refresh": "🔄 Обновить", "transcribe_btn": "📝 Распознать транскрипт",
    "seed": "Сид (-1 = случайно)", "max_tokens": "Макс. токенов / длительность",
    "examples": "Примеры", "tags_help": "❓ Все теги (подсказка)", "tags_legend": _LEGEND_RU,
    "ph_clone": "Текст, который произнесёт клонированный голос…",
    "cloud_title": "☁️ Скачать голоса с сервера (русский пак)", "cloud_status": "Статус",
    "load_list": "Обновить список", "cloud_voices": "Доступные голоса", "download_sel": "⬇️ Скачать выбранные",
    "refresh_voices": "🔄 Обновить список голосов",
    "num_speakers": "Количество дикторов", "pod_hint": "Опиши тему — режиссёр напишет диалог. Затем задай голоса дикторам и нажми «Озвучить».",
    "pod_format": _PODFMT_RU, "topic": "Тема подкаста", "ph_topic": "Напр.: плюсы и минусы локального ИИ дома",
    "make_script": "📝 Сгенерировать сценарий", "script": "Сценарий (можно править)",
    "ph_script": "Speaker 0: Привет!\nSpeaker 1: Здравствуй!", "synth": "🔊 Озвучить",
    "book_hint": "Вставь текст книги/главы, задай голоса (Speaker 0 — рассказчик), размечай по ролям и озвучивай.",
    "book_format": _BOOKFMT_RU, "book_text": "Текст книги / главы",
    "ph_book": "Вставь фрагмент с репликами персонажей…", "markup": "📝 Разметить по ролям",
    "batch_text": "Список текстов (по одному в строке)", "ph_batch": "Первая фраза.\nВторая фраза.\nТретья фраза.",
    "log": "Лог", "brand_header_html": BRAND_HTML_RU,
    "tab_long_clone": "🧬 Длинный клон",
    "max_chars_label": "Макс. длина фрагмента (символов)",
    "max_chars_info": "Оптимально: 150-250. Меньше — рваная интонация, больше — риск сбоев и повторов в конце фрагмента.",
    "gap_label": "Пауза между фрагментами (сек)",
    "merge_label": "Склеить в один аудиофайл",
    "long_clone_log": "Лог генерации",
    "llm_settings_title": "⚙️ Настройки внешнего API LLM и системного промпта",
    "api_url_label": "Адрес API (LM Studio / Ollama / OpenAI)",
    "api_key_label": "Ключ API (если требуется)",
    "api_model_label": "Имя модели в API",
    "system_prompt_label": "Системный промпт (правила тегов добавятся сами)",
    "save_settings_btn": "💾 Сохранить настройки LLM",
    "test_connection_btn": "🧪 Проверить подключение",
    "processed_text_label": "📝 Текст после LLM (режиссёра)",
    "cpu_only_label": "💻 Использовать только CPU (без видеокарты)",
    "keep_vram_label": "⚡ Не выгружать модели из памяти (ускоряет повторные запуски)",
    "lc_custom_num": "Кастомный номер",
    "lc_num_input": "Номер файла",
    "design_hint": "Опишите голос словами (возраст, тембр, эмоции, темп, акцент) и синтезируйте речь с нуля (FireRedTTS3).",
    "design_inst": "Описание голоса (Instruction)",
    "ph_design_inst": "напр.: Спокойный, глубокий мужской голос 35 лет с тёплым бархатным тембром, уверенный тон",
    "voice_plan_label": "План генерации голоса (Voice Plan / CoT)",
    "edit_hint": "Редактирование аудио (FireRedTTS3): замена/вставка слов (семантическое) или изменение скорости, тона, громкости (акустическое).",
    "edit_audio_label": "Исходное аудио",
    "edit_mode_label": "Режим редактирования",
    "mode_semantic": "Семантический (замена / вставка / удаление слов)",
    "mode_acoustic": "Акустический (скорость / тональность / громкость)",
    "sem_inst_label": "Инструкция изменения текста",
    "ph_sem_inst": "напр.: Replace 'cats' with 'dogs'",
    "ac_mode_label": "Параметр изменения",
    "ac_val_label": "Коэффициент",
    "custom_inst_label": "Своя инструкция (необязательно)",
    "edited_text_label": "Итоговый текст после правки",
    "example_btn": "💡 Пример текста",
}

_EN = {
    "engine_higgs_tab": "🎙️ Higgs Audio v3.1 (100+ Languages, Emotions & Director)",
    "engine_firered_tab": "🔥 FireRedTTS3 (24 Languages, Voice Design, Speech Edit)",
    "tab_tts": "🎙️ TTS", "tab_expr": "🎭 Expressive + Director", "tab_clone": "🧬 Cloning",
    "tab_design": "🎨 Voice Design", "tab_edit": "✂️ Speech Edit",
    "tab_pod": "🎬 Podcast", "tab_book": "📚 Audiobook", "tab_batch": "📦 Batch",
    "engine_label": "TTS Engine",
    "lang_label": "Language (for FireRedTTS3)",
    "text": "Text", "ph_text": "Type text…",
    "generate": "🔊 Generate", "stop": "⏹ Stop", "result": "Result", "advanced": "Advanced",
    "director_model": "Director model (enrich / dialogues)",
    "quant": "Quantization / Precision",
    "quant_info": "bf16 — best quality. 8-bit / int8 — saves VRAM while preserving fidelity. 4-bit — for low-VRAM GPUs.",
    "out_format": "Output format",
    "cat_emotion": "😊 Emotion (per sentence)", "cat_prosody": "🎵 Prosody", "cat_style": "🎭 Style", "cat_sfx": "🔊 Sounds (inline)",
    "download_all": "⬇️ Download all 700+",
    "enrich": "✨ Enrich text", "auto_enrich": "✨ Auto-enrich prompt with director",
    "upload_txt": "📂 Upload text from file (.txt)",
    "ref_voice": "Reference audio (voice)", "ref_text": "Reference transcript (auto-filled)",
    "ph_clone_tr": "What the reference says…", "voice_preset": "Voice preset",
    "refresh": "🔄 Refresh", "transcribe_btn": "📝 Transcribe reference",
    "seed": "Seed (-1 = random)", "max_tokens": "Max tokens / duration",
    "examples": "Examples", "tags_help": "❓ All tags (legend)", "tags_legend": _LEGEND_EN,
    "ph_clone": "Text the cloned voice will speak…",
    "cloud_title": "☁️ Download voices from server (Russian pack)", "cloud_status": "Status",
    "load_list": "Refresh list", "cloud_voices": "Available voices", "download_sel": "⬇️ Download selected",
    "refresh_voices": "🔄 Refresh voice list",
    "num_speakers": "Number of speakers", "pod_hint": "Describe a topic — the director writes a dialogue. Then set speaker voices and synthesize.",
    "pod_format": _PODFMT_EN, "topic": "Podcast topic", "ph_topic": "e.g. pros and cons of local AI at home",
    "make_script": "📝 Generate script", "script": "Script (editable)",
    "ph_script": "Speaker 0: Hello!\nSpeaker 1: Hi there!", "synth": "🔊 Synthesize",
    "book_hint": "Paste book/chapter text, set voices (Speaker 0 = narrator), attribute roles and synthesize.",
    "book_format": _BOOKFMT_EN, "book_text": "Book / chapter text",
    "ph_book": "Paste a passage with character dialogue…", "markup": "📝 Attribute roles",
    "batch_text": "List of texts (one per line)", "ph_batch": "First line.\nSecond line.\nThird line.",
    "log": "Log", "brand_header_html": BRAND_HTML_EN,
    "tab_long_clone": "🧬 Long Cloning",
    "max_chars_label": "Max fragment length (chars)",
    "max_chars_info": "Optimal: 150-250. Lower yields choppy phrasing, higher increases degradation/repetition risks.",
    "gap_label": "Pause between fragments (sec)",
    "merge_label": "Merge into single audio file",
    "long_clone_log": "Generation log",
    "llm_settings_title": "⚙️ External LLM API and System Prompt Settings",
    "api_url_label": "API URL (LM Studio / Ollama / OpenAI)",
    "api_key_label": "API Key (if required)",
    "api_model_label": "Model name in API",
    "system_prompt_label": "System prompt (tag rules will be appended automatically)",
    "save_settings_btn": "💾 Save LLM Settings",
    "test_connection_btn": "🧪 Test Connection",
    "processed_text_label": "📝 Text after LLM (director)",
    "cpu_only_label": "💻 Use CPU only (no GPU)",
    "keep_vram_label": "⚡ Keep models in VRAM (speeds up subsequent runs)",
    "lc_custom_num": "Custom number",
    "lc_num_input": "File number",
    "design_hint": "Describe a voice in natural language (age, timbre, accent, pace) and synthesize from scratch (FireRedTTS3).",
    "design_inst": "Voice description (Instruction)",
    "ph_design_inst": "e.g. A calm, deep male voice in his 30s with a warm baritone timbre",
    "voice_plan_label": "Voice Generation Plan (CoT)",
    "edit_hint": "Speech editing (FireRedTTS3): semantic word modification or acoustic speed/pitch/volume editing.",
    "edit_audio_label": "Input audio to edit",
    "edit_mode_label": "Edit mode",
    "mode_semantic": "Semantic (replace / insert / delete words)",
    "mode_acoustic": "Acoustic (speed / pitch / volume)",
    "sem_inst_label": "Edit instruction",
    "ph_sem_inst": "e.g. Replace 'cats' with 'dogs'",
    "ac_mode_label": "Parameter",
    "ac_val_label": "Value",
    "custom_inst_label": "Custom instruction override (optional)",
    "edited_text_label": "Rewritten transcript",
    "example_btn": "💡 Example Text",
}

I18N = gr.I18n(en=_EN, ru=_RU)


def T(key):
    return I18N(key)


HEAD_SCRIPT = """
<script>
(function(){
  var lang = null;
  try {
    lang = new URL(window.location).searchParams.get('__lang') || localStorage.getItem('gradio_lang') || 'ru';
  } catch(e) { lang = 'ru'; }

  try {
    localStorage.setItem('gradio_lang', lang);
    Object.defineProperty(navigator, 'language',  {get: function(){ return lang; }, configurable: true});
    Object.defineProperty(navigator, 'languages', {get: function(){ return [lang]; }, configurable: true});
    document.documentElement.lang = lang;
  } catch(e) {}

  window.__setLang = function(newLang) {
    try {
      localStorage.setItem('gradio_lang', newLang);
    } catch(e) {}
    var url = new URL(window.location.href);
    url.searchParams.set('__lang', newLang);
    window.location.href = url.toString();
  };

  var RU_TO_EN = {
    "Higgs Audio v3.1 (100+ языков, Эмоции & Режиссёр)": "Higgs Audio v3.1 (100+ Languages, Emotions & Director)",
    "FireRedTTS3 (24 языка, Voice Design, Редактирование речи)": "FireRedTTS3 (24 Languages, Voice Design, Speech Edit)",
    "Озвучка": "TTS",
    "Экспрессия + Режиссёр": "Expressive + Director",
    "Клонирование": "Cloning",
    "Создание голоса": "Voice Design",
    "Редактирование речи": "Speech Edit",
    "Длинный клон": "Long Cloning",
    "Подкаст": "Podcast",
    "Аудиокнига": "Audiobook",
    "Пакет": "Batch",
    "Использовать только CPU (без видеокарты)": "Use CPU only (no GPU)",
    "Не выгружать модели из памяти (ускоряет повторные запуски)": "Keep models in VRAM (speeds up subsequent runs)",
    "Формат вывода": "Output format",
    "Квантизация / Точность": "Quantization / Precision",
    "bf16 — высокое качество (рекомендуется)": "bf16 — high quality (recommended)",
    "8-bit / INT8 — экономия памяти": "8-bit / INT8 — saves VRAM",
    "4-bit — минимум VRAM (только Higgs)": "4-bit — minimum VRAM (Higgs only)",
    "fp32 — полная точность": "fp32 — full precision",
    "Язык (для FireRedTTS3)": "Language (for FireRedTTS3)",
    "Модель режиссёра (для обогащения / диалогов)": "Director model (for dialogues / enrich)",
    "Текст": "Text",
    "Введите текст…": "Type text…",
    "Загрузить текст из файла (.txt)": "Upload text (.txt)",
    "Пример текста": "Example text",
    "Пресет голоса": "Voice preset",
    "(По умолчанию / Default)": "(Default)",
    "— свой файл / own file —": "— own file —",
    "Обновить список голосов": "Refresh voice list",
    "Обновить список": "Refresh list",
    "Обновить": "Refresh",
    "Доп. настройки": "Advanced settings",
    "Озвучить": "Generate",
    "Стоп": "Stop",
    "Результат": "Result",
    "Авто-обогащение промпта режиссёром": "Auto-enrich prompt with director",
    "Обогатить текст": "Enrich text",
    "Аудио-референс (голос)": "Reference audio (voice)",
    "Транскрипт референса (заполнится сам)": "Reference transcript (auto-filled)",
    "Что произносится в референсе…": "What the reference voice says…",
    "Распознать транскрипт": "Transcribe reference",
    "Сид (-1 = случайно)": "Seed (-1 = random)",
    "Макс. токенов / длительность": "Max tokens / duration",
    "Примеры": "Examples",
    "Все теги (подсказка)": "All tags (legend / help)",
    "Скачать голоса с сервера (русский пак)": "Download cloud voices (Russian pack)",
    "Статус": "Status",
    "Доступные голоса": "Available voices",
    "Скачать выбранные": "Download selected",
    "Скачать все 700+": "Download all 700+",
    "Количество дикторов": "Number of speakers",
    "Тема подкаста": "Podcast topic",
    "Сгенерировать сценарий": "Generate script",
    "Сценарий (можно править)": "Script (editable)",
    "Разметить по ролям": "Attribute roles",
    "Текст книги / главы": "Book / chapter text",
    "Список текстов (по одному в строке)": "List of texts (one per line)",
    "Лог генерации": "Generation log",
    "Лог": "Log",
    "Макс. длина фрагмента (символов)": "Max chunk length (chars)",
    "Пауза между фрагментами (сек)": "Pause between chunks (sec)",
    "Склеить в один аудиофайл": "Merge into single audio file",
    "Кастомный номер": "Custom file number",
    "Номер файла": "File number",
    "Описание голоса (Instruction)": "Voice description (Instruction)",
    "План генерации голоса (Voice Plan / CoT)": "Voice Generation Plan (CoT)",
    "Исходное аудио": "Input audio",
    "Режим редактирования": "Edit mode",
    "Семантическое (слова)": "Semantic (words)",
    "Акустическое (скорость/тон/громкость)": "Acoustic (speed/pitch/volume)",
    "Семантический (замена / вставка / удаление слов)": "Semantic (replace / insert / delete words)",
    "Акустический (скорость / тональность / громкость)": "Acoustic (speed / pitch / volume)",
    "Инструкция изменения текста": "Text edit instruction",
    "Параметр изменения": "Parameter",
    "Коэффициент": "Value",
    "Своя инструкция (необязательно)": "Custom instruction override (optional)",
    "Итоговый текст после правки": "Rewritten transcript",
    "Настройки внешнего API LLM и системного промпта": "External LLM API and System Prompt Settings",
    "Адрес API (LM Studio / Ollama / OpenAI)": "API URL (LM Studio / Ollama / OpenAI)",
    "Ключ API (если требуется)": "API Key (if required)",
    "Имя модели в API": "Model name in API",
    "Системный промпт (правила тегов добавятся сами)": "System prompt (rules added automatically)",
    "Сохранить настройки LLM": "Save LLM Settings",
    "Проверить подключение": "Test Connection",
    "Текст после LLM (режиссёра)": "Text after LLM (director)",
    "Опишите голос словами (возраст, тембр, эмоции, темп, акцент) и синтезируйте речь с нуля (FireRedTTS3).": "Describe a voice in natural language (age, timbre, accent, pace) and synthesize from scratch (FireRedTTS3).",
    "Редактирование аудио (FireRedTTS3): замена/вставка слов (семантическое) или изменение скорости, тона, громкости (акустическое).": "Speech editing (FireRedTTS3): semantic word modification or acoustic speed/pitch/volume editing.",
    "Опиши тему — режиссёр напишет диалог. Затем задай голоса дикторам и нажми «Generate».": "Describe a topic — the director writes a dialogue. Then set speaker voices and synthesize.",
    "Опиши тему — режиссёр напишет диалог. Затем задай голоса дикторам и нажми «Озвучить».": "Describe a topic — the director writes a dialogue. Then set speaker voices and synthesize.",
    "Вставь текст книги/главы, задай голоса (Speaker 0 — рассказчик), размечай по ролям и озвучивай.": "Paste book/chapter text, set voices (Speaker 0 = narrator), attribute roles and synthesize.",
    "Формат сценария: каждая строка": "Script format: each line",
    "Формат сценария:": "Script format:",
    "каждая строка": "each line",
    "реплика": "line",
    "также Диктор N:, [N]": "also Speaker N:, [N]",
    "Номер = диктор ниже.": "The number = speaker below.",
    "Формат: Speaker 0: — рассказчик, Speaker 1+: — персонажи. Разметь кнопкой или вручную, потом озвучь.": "Format: Speaker 0: — narrator, Speaker 1+: — characters. Attribute roles with button or manually, then synthesize.",
    "Формат: Speaker 0: — рассказчик": "Format: Speaker 0: — narrator",
    "персонажи. Разметь кнопкой или вручную, потом озвучь.": "characters. Attribute roles with button or manually, then synthesize.",
    "Перетащите аудио сюда": "Drop audio here",
    "Нажмите для загрузки": "Click to upload",
    "Текст, который произнесёт клонированный голос…": "Text the cloned voice will speak…",
    "Что произносится в референсе…": "What the reference voice says…",
    "Вставь фрагмент с репликами персонажей…": "Paste a passage with character dialogue…",
    "Использовать через API": "Use via API",
    "Создано с помощью Gradio": "Built with Gradio",
    "Настройки": "Settings",
    "Привет! Это новая гибридная студия синтеза речи. Вы можете с...": "Hello! This is a modern hybrid speech synthesis studio...",
    "Привет! Это новая гибридная студия синтеза речи. Вы можете свободно переключаться между Higgs Audio v3 и FireRedTTS3.": "Hello! This is a modern hybrid speech synthesis studio. You can freely switch between Higgs Audio v3 and FireRedTTS3.",
    "Привет! Это новая гибридная студия синтеза речи.": "Hello! This is a modern hybrid speech synthesis studio.",
    "Привет! Это новая гибридная студия": "Hello! This is a modern hybrid speech studio",
    "Искусственный интеллект открывает совершенно новые горизонты...": "Artificial intelligence opens entirely new horizons...",
    "Искусственный интеллект открывает совершенно новые горизонты в создании музыки, подкастов и аудиокниг.": "Artificial intelligence opens entirely new horizons in music, podcast, and audiobook creation.",
    "Искусственный интеллект открывает совершенно новые горизонты": "Artificial intelligence opens entirely new horizons",
    "<|emotion:elation|>Ура! У нас всё получилось с первого раза! <|sfx:laughter|>ха-ха!": "<|emotion:elation|>Hooray! We got it right on the very first try! <|sfx:laughter|>ha-ha!",
    "<|emotion:elation|>Ура! У нас всё получилось с первого раза!...": "<|emotion:elation|>Hooray! We got it right on the very first try!...",
    "<|emotion:calm|>Вечерний город плавно погружался в мягкие сумерки.": "<|emotion:calm|>The evening city gently faded into a soft twilight.",
    "<|style:shouting|>Поднажми! Финиш уже совсем близко!": "<|style:shouting|>Keep going! The finish line is very close!",
    "Старый маяк молчал уже много лет. — Здесь кто-нибудь есть? — крикнул Том, поднимаясь по скрипучей лестнице. Ответом была лишь тишина.": "The old lighthouse had been silent for years. 'Is anyone here?' Tom shouted as he climbed the creaky stairs. Silence was the only answer.",
    "Старый маяк молчал уже много лет. — Здесь кто-нибудь есть?...": "The old lighthouse had been silent for years. 'Is anyone here?'...",
    "Старый маяк молчал уже много лет.": "The old lighthouse had been silent for years.",
    "Плюсы и минусы локального ИИ дома": "Pros and cons of local AI at home",
    "Как нейросети меняют музыку": "How neural networks are changing music",
    "Будущее голосовых ассистентов": "The future of voice assistants"
  };

  var RU_KEYS = Object.keys(RU_TO_EN).sort(function(a, b) { return b.length - a.length; });

  function translateDom(root) {
    if (lang !== 'en') return;
    try {
      var walker = document.createTreeWalker(root || document.body, NodeFilter.SHOW_TEXT, null, false);
      var node;
      while (node = walker.nextNode()) {
        var val = node.nodeValue;
        if (!val || !val.trim()) continue;
        for (var i = 0; i < RU_KEYS.length; i++) {
          var k = RU_KEYS[i];
          if (val.indexOf(k) !== -1) {
            val = val.split(k).join(RU_TO_EN[k]);
          }
        }
        if (val !== node.nodeValue) {
          node.nodeValue = val;
        }
      }
      var inputs = (root || document).querySelectorAll('input, textarea');
      inputs.forEach(function(inp) {
        if (inp.placeholder) {
          var p = inp.placeholder;
          for (var i = 0; i < RU_KEYS.length; i++) {
            var k = RU_KEYS[i];
            if (p.indexOf(k) !== -1) p = p.split(k).join(RU_TO_EN[k]);
          }
          inp.placeholder = p;
        }
        if (inp.value) {
          var v = inp.value;
          for (var i = 0; i < RU_KEYS.length; i++) {
            var k = RU_KEYS[i];
            if (v.indexOf(k) !== -1) v = v.split(k).join(RU_TO_EN[k]);
          }
          if (v !== inp.value) inp.value = v;
        }
      });
      var cells = (root || document).querySelectorAll('table td, table th, .dataset button, .gr-dataset');
      cells.forEach(function(c) {
        if (c.children.length === 0 && c.textContent) {
          var t = c.textContent;
          for (var i = 0; i < RU_KEYS.length; i++) {
            var k = RU_KEYS[i];
            if (t.indexOf(k) !== -1) {
              c.textContent = t.split(k).join(RU_TO_EN[k]);
              break;
            }
          }
        }
      });
    } catch(e) {}
  }

  // Ensure all tab buttons in sub-tabs stay unhidden on initial load, tab change, and resize
  function ensureAllTabsVisible() {
    document.querySelectorAll('.sub-tabs [role="tablist"] > button, .sub-tabs .tab-nav > button').forEach(function(b) {
      if (b.classList.contains('overflow-menu-button') || b.getAttribute('aria-label') === 'More' || b.getAttribute('aria-haspopup') === 'menu' || b.textContent.trim() === '…' || b.textContent.trim() === '...') {
        b.style.setProperty('display', 'none', 'important');
      } else {
        if (b.hasAttribute('hidden')) b.removeAttribute('hidden');
        if (b.style.display === 'none') b.style.setProperty('display', 'inline-flex', 'important');
      }
    });
    translateDom();
  }

  setTimeout(ensureAllTabsVisible, 100);
  setTimeout(ensureAllTabsVisible, 300);
  setTimeout(ensureAllTabsVisible, 800);
  setTimeout(ensureAllTabsVisible, 1500);
  window.addEventListener('resize', ensureAllTabsVisible);
  window.addEventListener('click', function(){ setTimeout(ensureAllTabsVisible, 50); });

  try {
    var tabObserver = new MutationObserver(function() { ensureAllTabsVisible(); });
    tabObserver.observe(document.documentElement, { attributes: true, subtree: true, attributeFilter: ['hidden', 'style', 'class'], childList: true });
  } catch(e) {}
})();
</script>
"""

CSS = """
.gradio-container { width: 100% !important; max-width: 1180px !important; min-width: 900px !important; margin: 0 auto !important; box-sizing: border-box !important; }
.brand-header { position: relative; }
.brand-box { background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #7e22ce 100%);
  padding: 22px 26px; border-radius: 16px; margin: 6px 0 14px 0;
  box-shadow: 0 10px 30px rgba(109,40,217,0.35); color: white; text-align: center; }
.brand-title { font-size: 1.85em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { font-size: 0.98em; opacity: 0.9; margin-bottom: 12px; }
.brand-credits { font-size: 0.88em; opacity: 0.95; }
.device-badge { display:inline-block; background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:999px; font-size:0.85em; margin-top:8px; }
.lang-switcher { position:absolute; top:12px; right:16px; display:flex; gap:6px; align-items:flex-start; z-index:50; }
.lang-btn { background:rgba(255,255,255,0.18); color:white !important; padding:5px 10px; border-radius:8px;
  font-size:0.82em; text-decoration:none !important; font-weight:600; display:inline-flex; flex-direction:column;
  align-items:center; justify-content:center; gap:2px; line-height:1; white-space:nowrap; min-width:44px; cursor:pointer; }
.lang-btn:hover { background:rgba(255,255,255,0.3); }
.lang-btn img { margin:0 !important; vertical-align:middle !important; }

/* Top-level Engine Tabs */
.engine-main-tabs > div[role="tablist"], .engine-main-tabs > .tab-nav {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  gap: 12px !important;
  margin-bottom: 16px !important;
  border-bottom: 2px solid rgba(255,255,255,0.10) !important;
  padding-bottom: 6px !important;
}
.engine-main-tabs > div[role="tablist"] > button, .engine-main-tabs > .tab-nav > button {
  flex: 1 1 50% !important;
  font-size: 14.5px !important;
  font-weight: 700 !important;
  padding: 10px 18px !important;
  min-height: 44px !important;
  border-radius: 10px !important;
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: #e2e8f0 !important;
  text-align: center !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}
.engine-main-tabs > div[role="tablist"] > button:hover, .engine-main-tabs > .tab-nav > button:hover {
  background: rgba(124,58,237,0.18) !important;
  border-color: rgba(124,58,237,0.5) !important;
}
.engine-main-tabs > div[role="tablist"] > button.selected, .engine-main-tabs > .tab-nav > button.selected {
  background: linear-gradient(135deg, #581c87 0%, #7c3aed 100%) !important;
  color: #ffffff !important;
  border-color: #a78bfa !important;
  box-shadow: 0 4px 20px rgba(124,58,237,0.45) !important;
}

/* Engine Dropdown Row (Language / Director model) */
.engine-dropdown-row {
  margin-top: 10px !important;
  margin-bottom: 18px !important;
  clear: both !important;
  position: relative !important;
}

/* Sub-tabs container */
.sub-tabs {
  margin-top: 12px !important;
  clear: both !important;
  position: relative !important;
}

/* Sub-tabs inside each engine: clean 4-column grid (exactly 2 rows), compact size, NO overlapping */
.sub-tabs .tab-nav,
.sub-tabs div[role="tablist"],
.tabs.sub-tabs > div.tab-nav,
.sub-tabs [role="tablist"] {
  display: grid !important;
  grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  gap: 8px !important;
  border-bottom: 2px solid rgba(255,255,255,0.08) !important;
  padding: 4px 0 16px 0 !important;
  margin: 8px 0 24px 0 !important;
  height: auto !important;
  min-height: 0 !important;
  max-height: none !important;
  overflow: visible !important;
  width: 100% !important;
  max-width: 100% !important;
  position: relative !important;
  clear: both !important;
}

.sub-tabs .tab-nav > button,
.sub-tabs div[role="tablist"] > button,
.tabs.sub-tabs > div.tab-nav > button,
.sub-tabs [role="tablist"] > button {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  height: 38px !important;
  min-height: 38px !important;
  max-height: 38px !important;
  box-sizing: border-box !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: center !important;
  justify-content: center !important;
  align-items: center !important;
  padding: 4px 8px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: #cbd5e1 !important;
  display: inline-flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  margin: 0 !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
}

.sub-tabs .tab-nav > button[hidden],
.sub-tabs div[role="tablist"] > button[hidden],
.sub-tabs [role="tablist"] > button[hidden],
.sub-tabs .tab-nav > button[style*="none"],
.sub-tabs div[role="tablist"] > button[style*="none"],
.sub-tabs [role="tablist"] > button[style*="none"] {
  display: inline-flex !important;
  visibility: visible !important;
  opacity: 1 !important;
}

.sub-tabs .tab-nav > button:hover,
.sub-tabs div[role="tablist"] > button:hover,
.sub-tabs [role="tablist"] > button:hover {
  background: rgba(124,58,237,0.15) !important;
  border-color: rgba(124,58,237,0.35) !important;
  color: #ffffff !important;
}

.sub-tabs .tab-nav > button.selected,
.sub-tabs div[role="tablist"] > button.selected,
.sub-tabs [role="tablist"] > button.selected {
  background: #7c3aed !important;
  color: #ffffff !important;
  border-color: #9333ea !important;
  box-shadow: 0 3px 10px rgba(124,58,237,0.35) !important;
}

/* Ensure tab contents / panels have clear, spacious margin below the buttons */
.sub-tabs > div[role="tabpanel"],
.sub-tabs .tabitem,
.tabs.sub-tabs > div:not(.tab-nav) {
  margin-top: 24px !important;
  padding-top: 4px !important;
  clear: both !important;
  position: relative !important;
}

/* Completely suppress and hide Gradio's overflow menu button (...) */
.sub-tabs .overflow-menu,
.sub-tabs .overflow-menu-button,
.sub-tabs [aria-label="More"],
.sub-tabs [aria-haspopup="menu"],
.sub-tabs div.overflow-menu-wrapper {
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  height: 0 !important;
  pointer-events: none !important;
  opacity: 0 !important;
}

.spk-block { background: rgba(124,58,237,0.06); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:10px; margin:6px 0; }
.gradio-container { --block-border-color: rgba(255,255,255,0.10) !important;
  --border-color-primary: rgba(255,255,255,0.10) !important;
  --input-border-color: rgba(255,255,255,0.10) !important;
  --neutral-200: rgba(255,255,255,0.10) !important; }
.gradio-container .block { border-color: rgba(255,255,255,0.10) !important; }
.scrollable-files { max-height: 280px; overflow-y: auto !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px; }

.fmt-radio-row .wrap,
.fmt-radio-row div[role="radiogroup"],
.fmt-radio-row .gradio-radio-group {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  gap: 14px !important;
  align-items: center !important;
  min-height: 42px !important;
}
.fmt-radio-row label {
  margin-bottom: 0 !important;
}
"""

DARK_JS = """
function() {
  document.body.classList.add('dark');
}
"""

TAG_RU = {
    "happy": "счастье", "sad": "грусть", "angry": "злость", "fearful": "страх",
    "disgusted": "отвращение", "surprised": "удивление", "neutral": "нейтрально",
    "excited": "воодушевление", "anxious": "тревога", "confused": "замешательство",
    "calm": "спокойствие", "proud": "гордость", "guilty": "вина", "ashamed": "стыд",
    "lonely": "одиночество", "jealous": "зависть", "hopeful": "надежда", "grateful": "благодарность",
    "embarrassed": "смущение", "relieved": "облегчение", "elation": "восторг",
    "fast": "быстро", "slow": "медленно", "loud": "громко", "soft": "тихо",
    "high_pitch": "высокий тон", "low_pitch": "низкий тон", "whisper": "шёпот",
    "shout": "крик", "sing": "пение", "monotone": "монотонно", "shouting": "кричащий",
    "casual": "разговорный", "formal": "деловой", "laughter": "смех", "crying": "плач",
    "sigh": "вздох", "gasp": "вдох от неожиданности", "throat_clearing": "покашливание",
    "yawn": "зевок", "cough": "кашель", "sneeze": "чихание", "applause": "аплодисменты",
}

TAGS_LEGEND_MD = f"""
### Легенда тегов экспрессии (Higgs Audio v3)

{_LEGEND_RU}
"""

TTS_EXAMPLES = [
    ["Привет! Это новая гибридная студия синтеза речи. Вы можете свободно переключаться между Higgs Audio v3 и FireRedTTS3."],
    ["Искусственный интеллект открывает совершенно новые горизонты в создании музыки, подкастов и аудиокниг."],
]

EXPR_EXAMPLES = [
    ["<|emotion:elation|>Ура! У нас всё получилось с первого раза! <|sfx:laughter|>ха-ха!"],
    ["<|emotion:calm|>Вечерний город плавно погружался в мягкие сумерки."],
    ["<|style:shouting|>Поднажми! Финиш уже совсем близко!"],
]
POD_TOPICS = [["Плюсы и минусы локального ИИ дома"], ["Как нейросети меняют музыку"],
              ["Будущее голосовых ассистентов"]]
BOOK_EXAMPLES = [["Старый маяк молчал уже много лет. — Здесь кто-нибудь есть? — крикнул Том, "
                  "поднимаясь по скрипучей лестнице. Ответом была лишь тишина."]]


# ----------------------------------------------------------------------------
# Хелперы
# ----------------------------------------------------------------------------
_OUT_FORMAT = "mp3"
_FMT = {"wav": ("WAV", None), "mp3": ("MP3", None), "flac": ("FLAC", None), "ogg": ("OGG", "VORBIS")}


def set_out_format(f):
    global _OUT_FORMAT
    _OUT_FORMAT = f if f in _FMT else "wav"


def _save(sr, wav, prefix="tts"):
    import soundfile as sf
    if wav is None or len(wav) == 0:
        return None
    fmt = _OUT_FORMAT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container, subtype = _FMT.get(fmt, ("WAV", None))
    path = OUTPUT_DIR / f"{prefix}_{stamp}.{fmt}"
    try:
        sf.write(str(path), wav, sr, format=container, subtype=subtype)
    except Exception as e:
        print(f"[save] формат {fmt} не записался ({e}) → wav")
        path = OUTPUT_DIR / f"{prefix}_{stamp}.wav"
        sf.write(str(path), wav, sr)
    return str(path)


def scan_voices():
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted(p.stem for p in VOICES_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts)


def voice_path(name):
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            return str(p)
    return None


def voice_transcript(name):
    for ext in (".txt", ".lab"):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            for enc in ("utf-8", "cp1251"):
                try:
                    return p.read_text(encoding=enc).strip()
                except Exception:
                    continue
    return ""


def cb_preset(name):
    if not name or name == OWN_FILE:
        return None, ""
    return voice_path(name), voice_transcript(name)


_ASR = None


def _get_asr():
    global _ASR
    if _ASR is None:
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        proc = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
        amodel = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base").eval()
        _ASR = (proc, amodel)
    return _ASR


def transcribe(ref_audio):
    if not ref_audio:
        return gr.update()
    if eng._MOCK:
        return "пример транскрипта (mock)"
    try:
        import torch
        import soundfile as sf
        import torchaudio
        proc, amodel = _get_asr()
        data, sr = sf.read(ref_audio, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        inp = proc(wav.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            tok = amodel.generate(**inp)
        return proc.decode(tok[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[asr] {e}")
        return gr.update()


_SPK_PATTERNS = [r'^speaker\s*(\d+)\s*:\s*(.+)$', r'^диктор\s*(\d+)\s*:\s*(.+)$',
                 r'^голос\s*(\d+)\s*:\s*(.+)$', r'^\[(\d+)\]\s*(.+)$']


def parse_script(script):
    out = []
    for line in (script or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        matched = False
        for pat in _SPK_PATTERNS:
            m = re.match(pat, line, re.IGNORECASE)
            if m:
                out.append((int(m.group(1)), m.group(2).strip()))
                matched = True
                break
        if not matched:
            out.append((0, line))
    return out


def _chunk(text, max_chars=120):
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        cur = ""
        for s in re.split(r"(?<=[.!?…])\s+", para):
            if cur and len(cur) + len(s) > max_chars:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            chunks.append(cur)
    return chunks


def _speak(text, ref_audio=None, ref_text=None, engine_choice=None, language="auto", **kw):
    """Маршрутизация синтеза на активный движок (Higgs v3.1 или FireRedTTS3)."""
    is_firered = bool(engine_choice and "FireRed" in engine_choice)
    if is_firered:
        import re
        clean_text = re.sub(r"<\|[^>]+:.*?\|>", "", text).strip()
        return fe.generate(clean_text, ref_audio=ref_audio, ref_text=ref_text, language=language, **kw)
    else:
        chunks = _chunk(text)
        if len(chunks) <= 1:
            return eng.generate(text, ref_audio=ref_audio, ref_text=ref_text, **kw)
        return eng.synth_longform(chunks, ref_audio=ref_audio, ref_text=ref_text, **kw)


def cb_load_cloud():
    voices = []
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset"))
        voices = sorted(f[:-4] for f in files if f.endswith(".mp3"))
    except Exception as e:
        print(f"[voices] list: {e}")
    status = f"Найдено / Found: {len(voices)}" if voices else "Не удалось загрузить / Failed"
    return status, gr.update(choices=voices, value=[])


def _dl_voice(name):
    import requests
    try:
        r = requests.get(f"{CLOUD_VOICES_BASE}/{name}.mp3?download=true", timeout=90)
        r.raise_for_status()
        (VOICES_DIR / f"{name}.mp3").write_bytes(r.content)
        try:
            rt = requests.get(f"{CLOUD_VOICES_BASE}/{name}.txt?download=true", timeout=30)
            if rt.status_code == 200:
                (VOICES_DIR / f"{name}.txt").write_text(rt.text, encoding="utf-8")
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"[voices] dl {name}: {e}")
        return False


def cb_download_voices(selected):
    if not selected:
        return "Выберите голоса / Select voices", gr.update()
    ok = sum(_dl_voice(n) for n in selected)
    return f"Скачано / Downloaded: {ok}/{len(selected)}", gr.update(choices=[OWN_FILE] + scan_voices())


def cb_download_all_cloud(progress=gr.Progress()):
    try:
        from huggingface_hub import list_repo_files
        names = sorted(f[:-4] for f in list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset") if f.endswith(".mp3"))
    except Exception as e:
        return f"Ошибка списка / List error: {e}", gr.update()
    if not names:
        return "Список пуст / Empty list", gr.update()
    ok = 0
    for i, name in enumerate(names):
        progress((i + 1) / len(names), desc=f"{i + 1}/{len(names)} · {name}")
        if _dl_voice(name):
            ok += 1
    return f"Скачано / Downloaded: {ok}/{len(names)}", gr.update(choices=[OWN_FILE] + scan_voices())


# ----------------------------------------------------------------------------
# Колбэки
# ----------------------------------------------------------------------------
GUI_CONFIG_PATH = SCRIPT_DIR / "gui_config.json"


def load_gui_config():
    if GUI_CONFIG_PATH.exists():
        try:
            import json
            with open(GUI_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if cfg.get("model_dd") == "Внешнее API (LM Studio / Ollama / OpenAI)":
                    cfg["model_dd"] = "External API (LM Studio / Ollama / OpenAI)"
                return cfg
        except Exception as e:
            print(f"Error loading gui_config: {e}")
    return {}


def update_gui_config(key, val):
    try:
        import json
        config = load_gui_config()
        config[key] = val
        with open(GUI_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving gui_config: {e}")


def save_tts_settings(text, temp, top_p, top_k, max_new, seed, engine, lang, preset=None):
    update_gui_config("t_text", text)
    update_gui_config("t_temp", temp)
    update_gui_config("t_top_p", top_p)
    update_gui_config("t_top_k", top_k)
    update_gui_config("t_max", max_new)
    update_gui_config("t_seed", seed)
    update_gui_config("engine_choice", engine)
    update_gui_config("language_choice", lang)
    if preset:
        update_gui_config("t_preset", preset)


def save_expr_settings(text, auto):
    update_gui_config("e_text", text)
    update_gui_config("e_auto", auto)


def save_clone_settings(text, preset, auto, temp, top_p, seed, engine, lang):
    update_gui_config("c_text", text)
    update_gui_config("c_preset", preset)
    update_gui_config("c_auto", auto)
    update_gui_config("c_temp", temp)
    update_gui_config("c_top_p", top_p)
    update_gui_config("c_seed", seed)
    update_gui_config("engine_choice", engine)
    update_gui_config("language_choice", lang)


def save_long_clone_settings(text, preset, max_chars, gap, merge, temp, top_p, seed, auto, custom_num, num_input, engine, lang):
    update_gui_config("lc_text", text)
    update_gui_config("lc_preset", preset)
    update_gui_config("lc_max_chars", max_chars)
    update_gui_config("lc_gap", gap)
    update_gui_config("lc_merge", merge)
    update_gui_config("lc_temp", temp)
    update_gui_config("lc_top_p", top_p)
    update_gui_config("lc_seed", seed)
    update_gui_config("lc_auto", auto)
    update_gui_config("lc_custom_num", custom_num)
    update_gui_config("lc_num_input", num_input)
    update_gui_config("engine_choice", engine)
    update_gui_config("language_choice", lang)


def save_book_settings(text):
    update_gui_config("b_text", text)


def save_batch_settings(text, auto):
    update_gui_config("bt_text", text)
    update_gui_config("bt_auto", auto)


def load_text_file(temp_file):
    if temp_file is None:
        return ""
    try:
        path = temp_file.name if hasattr(temp_file, 'name') else temp_file
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


def _maybe_enrich(text, model, auto):
    if auto:
        try:
            res = dr.enrich(text, model)
            dr.unload_llm(force=False)
            if res and res.strip():
                return res
            print("[director] Enrich returned empty result. Fallback to original text.")
        except Exception as e:
            print(f"[director] Enrich error: {e}. Fallback to original text.")
    return text


def cb_tts(text, model, auto, temperature, top_p, top_k, max_new, seed, engine, lang, preset=None, progress=gr.Progress(track_tqdm=True)):
    eng.clear_cancel()
    fe.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    ref_a, ref_t = None, None
    if preset and preset not in ("(По умолчанию / Default)", "(Свой файл / Own file)", OWN_FILE):
        ref_a = voice_path(preset)
        ref_t = voice_transcript(preset)
    sr, wav = _speak(text, ref_audio=ref_a, ref_text=ref_t, temperature=temperature, top_p=top_p, top_k=top_k,
                     max_new_tokens=max_new, seed=seed, engine_choice=engine, language=lang)
    p = _save(sr, wav, "tts")
    return p, text


def cb_enrich(text, model):
    return dr.enrich(text, label=model)


def cb_expr(text, model, auto, engine, lang, progress=gr.Progress(track_tqdm=True)):
    eng.clear_cancel()
    fe.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    sr, wav = _speak(text, engine_choice=engine, language=lang)
    p = _save(sr, wav, "expr")
    return p, text


def cb_clone(text, model, auto, ref_audio, ref_text, preset, temperature, top_p, seed, engine, lang, progress=gr.Progress(track_tqdm=True)):
    eng.clear_cancel()
    fe.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    ref = ref_audio or (voice_path(preset) if preset and preset != OWN_FILE else None)
    sr, wav = _speak(text, ref_audio=ref, ref_text=ref_text, temperature=temperature,
                     top_p=top_p, seed=seed, engine_choice=engine, language=lang)
    p = _save(sr, wav, "clone")
    return p, text


def cb_voice_design(instruction, text, language, temp, top_p, top_k, rep_pen, timesteps, cfg, seed, max_sec, do_tn, do_split, cross_fade, progress=gr.Progress(track_tqdm=True)):
    fe.clear_cancel()
    sr, wav, plan = fe.voice_design(
        instruction=instruction,
        text=text,
        language=language,
        text_temperature=temp,
        text_top_p=top_p,
        text_top_k=top_k,
        text_repetition_penalty=rep_pen,
        n_timesteps=int(timesteps),
        inference_cfg=cfg,
        seed=seed,
        max_audio_seconds=max_sec,
        do_tn=do_tn,
        do_split=do_split,
        cross_fade_ms=cross_fade,
    )
    p = _save(sr, wav, "vdesign")
    return p, plan


def cb_semantic_edit(audio_path, instruction, timesteps, cfg, stop_th, seed, max_sec, progress=gr.Progress(track_tqdm=True)):
    fe.clear_cancel()
    sr, wav, edited_text = fe.semantic_edit(
        audio_path=audio_path,
        instruction=instruction,
        n_timesteps=int(timesteps),
        inference_cfg=cfg,
        stop_threshold=stop_th,
        seed=seed,
        max_audio_seconds=max_sec,
    )
    p = _save(sr, wav, "s_edit")
    return p, edited_text


def cb_acoustic_edit(audio_path, mode, value, custom_inst, timesteps, cfg, stop_th, seed, max_sec, progress=gr.Progress(track_tqdm=True)):
    fe.clear_cancel()
    sr, wav = fe.acoustic_edit(
        audio_path=audio_path,
        mode=mode,
        value=value,
        custom_instruction=custom_inst,
        n_timesteps=int(timesteps),
        inference_cfg=cfg,
        stop_threshold=stop_th,
        seed=seed,
        max_audio_seconds=max_sec,
    )
    p = _save(sr, wav, "a_edit")
    return p


def cb_test_llm_connection(api_url, api_key, api_model):
    import urllib.request
    import urllib.error
    import json

    url = (api_url or "http://127.0.0.1:1234/v1").rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, respond with exactly 'OK'"}
        ],
        "temperature": 0.3,
        "max_tokens": 10,
        "stream": False
    }
    if api_model:
        data["model"] = api_model

    req_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    try:
        print(f"[director] Testing LLM API connection to: {url} (model: {api_model})", flush=True)
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = response.read().decode("utf-8")
            res = json.loads(res_data)
            content = res["choices"][0]["message"]["content"].strip()
            print(f"[director] Connection test SUCCESS! Model responded: '{content}'", flush=True)
            return f"🟢 Успешно подключено! / Success! Response: '{content}'"
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            detail = json.loads(err_body)
            err_msg = detail.get("error", {}).get("message", str(e))
        except Exception:
            err_msg = str(e)
        err = f"🔴 Ошибка HTTP {e.code}: {err_msg}"
        print(f"[director] Connection test FAILED: {err}", flush=True)
        return err
    except urllib.error.URLError as e:
        err = f"🔴 Ошибка подключения: {e.reason}. Убедитесь, что LM Studio / Ollama запущены и порт указан верно."
        print(f"[director] Connection test FAILED: {err}", flush=True)
        return err
    except Exception as e:
        err = f"🔴 Ошибка: {e}"
        print(f"[director] Connection test FAILED: {err}", flush=True)
        return err


def cb_long_clone(text, model, auto, ref_audio, ref_text, preset, temperature, top_p, seed, max_chars, gap, merge,
                  api_url, api_key, api_model, system_prompt, custom_num, num_input, engine, lang, progress=gr.Progress(track_tqdm=True)):
    eng.clear_cancel()
    fe.clear_cancel()
    import soundfile as sf
    import numpy as np

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"long_clone_{stamp}"
    out_dir.mkdir(exist_ok=True)

    ref = ref_audio or (voice_path(preset) if preset and preset != OWN_FILE else None)

    num_str = (num_input or "").strip()
    if not num_str:
        num_str = "01"

    width = len(num_str)
    try:
        val = int(num_str)
        next_val = val + 1
        next_num_str = f"{next_val:0{width}d}"
    except ValueError:
        next_num_str = num_str

    chunks = _chunk(text, max_chars=int(max_chars))
    if not chunks:
        yield "Текст пуст / Text is empty", None, [], "", num_str
        return

    device, name, vram = eng.detect_device()
    log_lines = [
        f"Движок: {engine} | Устройство: {name}",
        f"Разбито на {len(chunks)} частей. Папка: output/long_clone_{stamp}"
    ]
    if custom_num:
        fmt = _OUT_FORMAT
        target_name = f"{num_str}.{fmt}"
        num_dir = OUTPUT_DIR / "NUM"
        target_path = num_dir / target_name
        if target_path.exists():
            warn_msg = f"⚠️ Предупреждение: файл {target_name} уже существует в папке NUM! Он будет перезаписан."
            gr.Warning(warn_msg)
            log_lines.insert(0, warn_msg)

    yield "\n".join(log_lines), None, [], "", num_str

    generated_files = []
    wav_chunks = []
    processed_chunks = []
    sr = 24000

    for i, chunk in enumerate(chunks):
        if eng.cancelled() or fe.cancelled():
            log_lines.append("\n⏹ Остановлено / Stopped.")
            yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks), num_str
            return

        progress((i + 1) / max(len(chunks), 1), desc=f"Фрагмент {i + 1}/{len(chunks)}")
        log_lines.append(f"✓ Синтез части {i + 1}/{len(chunks)}: '{chunk[:40]}...'")
        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks), num_str

        if auto:
            s_prompt = system_prompt + "\n" + dr._TAG_RULES
            try:
                if model == "External API (LM Studio / Ollama / OpenAI)":
                    enriched_chunk = dr.filter_tags(dr._chat(
                        s_prompt, chunk, label=model, temp=0.4,
                        api_url=api_url, api_key=api_key, api_model=api_model
                    ))
                else:
                    enriched_chunk = dr.filter_tags(dr._chat(
                        s_prompt, chunk, label=model, temp=0.4
                    ))
                chunk_to_speak = enriched_chunk.strip() if enriched_chunk and enriched_chunk.strip() else chunk
            except Exception as e:
                log_lines.append(f"  -> Ошибка LLM-режиссёра: {e}. Используем исходный текст.")
                chunk_to_speak = chunk
        else:
            chunk_to_speak = chunk

        processed_chunks.append(chunk_to_speak)

        try:
            chunk_sr, chunk_wav = _speak(
                chunk_to_speak,
                ref_audio=ref,
                ref_text=ref_text,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                engine_choice=engine,
                language=lang,
            )
            sr = chunk_sr

            if chunk_wav is not None and len(chunk_wav) > 0:
                filename = out_dir / f"chunk_{i + 1:03d}.wav"
                sf.write(str(filename), chunk_wav, sr)
                generated_files.append(str(filename))
                wav_chunks.append(chunk_wav)
                log_lines.append(f"  -> Сохранено: {filename.name}")
            else:
                log_lines.append(f"  -> Ошибка: пустой аудиопоток")
        except Exception as e:
            log_lines.append(f"  -> Ошибка генерации: {e}")

        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks), num_str

    final_wav = None
    final_saved_path = None

    if merge and wav_chunks and not (eng.cancelled() or fe.cancelled()):
        log_lines.append("\nСклеивание фрагментов...")
        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks), num_str

        silence_len = int(gap * sr)
        sil = np.zeros(silence_len, dtype=np.float32)

        out_wav = []
        for idx, w in enumerate(wav_chunks):
            if idx > 0:
                out_wav.append(sil)
            out_wav.append(w)

        merged_wav = np.concatenate(out_wav)
        merged_wav = eng._peak_limit(merged_wav)

        merged_filename = out_dir / f"merged_{stamp}.wav"
        sf.write(str(merged_filename), merged_wav, sr)

        _save_path = _save(sr, merged_wav, f"long_clone_merged")
        if _save_path:
            generated_files.insert(0, _save_path)

        log_lines.append(f"Склеено и сохранено: {merged_filename.name}")
        final_wav = merged_wav
        final_saved_path = _save_path
        yield "\n".join(log_lines), _save_path, generated_files, "\n\n".join(processed_chunks), num_str
    elif wav_chunks and not (eng.cancelled() or fe.cancelled()):
        final_wav = wav_chunks[0] if len(wav_chunks) == 1 else np.concatenate(wav_chunks)
        final_saved_path = generated_files[0] if generated_files else None
        yield "\n".join(log_lines), final_saved_path, generated_files, "\n\n".join(processed_chunks), num_str
    else:
        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks), num_str

    if custom_num and final_wav is not None and not (eng.cancelled() or fe.cancelled()):
        num_dir = OUTPUT_DIR / "NUM"
        num_dir.mkdir(parents=True, exist_ok=True)
        fmt = _OUT_FORMAT
        container, subtype = _FMT.get(fmt, ("WAV", None))
        custom_filepath = num_dir / f"{num_str}.{fmt}"

        try:
            sf.write(str(custom_filepath), final_wav, sr, format=container, subtype=subtype)
            custom_saved_path = str(custom_filepath)
            log_lines.append(f"✓ Сохранено в NUM: {custom_filepath.name}")
        except Exception as e:
            print(f"[save] custom num format {fmt} failed ({e}) -> wav")
            custom_filepath = num_dir / f"{num_str}.wav"
            sf.write(str(custom_filepath), final_wav, sr)
            custom_saved_path = str(custom_filepath)
            log_lines.append(f"✓ Сохранено в NUM: {custom_filepath.name}")

        if custom_saved_path:
            generated_files.insert(0, custom_saved_path)
            final_saved_path = custom_saved_path

    log_lines.append("\n🎉 Готово / Done!")
    final_num_str = next_num_str if (custom_num and not (eng.cancelled() or fe.cancelled()) and wav_chunks) else num_str
    yield "\n".join(log_lines), final_saved_path, generated_files, "\n\n".join(processed_chunks), final_num_str


def cb_podcast_script(topic, num, model):
    return dr.write_podcast(topic, int(num), label=model)


def cb_book_markup(text, num, model):
    return dr.cast_audiobook(text, int(num), label=model)


def cb_multi_synth(script, a0, a1, a2, a3, t0, t1, t2, t3, engine, lang, progress=gr.Progress()):
    eng.clear_cancel()
    fe.clear_cancel()
    audios = [a0, a1, a2, a3]
    texts = [t0, t1, t2, t3]
    turns = [(sid, txt) for sid, txt in parse_script(script) if txt.strip()]
    if not turns:
        return None
    chunks = []
    for i, (sid, txt) in enumerate(turns):
        if eng.cancelled() or fe.cancelled():
            break
        progress((i + 1) / len(turns), desc=f"{i + 1}/{len(turns)} · Speaker {sid}")
        ref = audios[sid] if 0 <= sid < MAX_SPK else None
        rt = (texts[sid] if 0 <= sid < MAX_SPK else None) or None
        _, wav = _speak(txt, ref_audio=ref, ref_text=rt, engine_choice=engine, language=lang)
        if wav is not None and len(wav):
            chunks.append(wav)
    final = eng._concat(chunks, gap=0.3)
    p = _save(eng.SR, final, "multi")
    return p


def cb_batch(texts, model, auto, engine, lang, progress=gr.Progress()):
    eng.clear_cancel()
    fe.clear_cancel()
    lines = [t.strip() for t in (texts or "").splitlines() if t.strip()]
    device, name, vram = eng.detect_device()
    log = [f"Движок: {engine} | Устройство: {name}", f"Начало пакетной обработки ({len(lines)} строк)"]
    paths = []
    for i, line in enumerate(lines):
        if eng.cancelled() or fe.cancelled():
            yield "\n".join(log) + "\n\n⏹ Остановлено / Stopped.", paths
            return
        progress((i + 1) / max(len(lines), 1), desc=f"{i + 1}/{len(lines)}")
        if auto:
            line = dr.enrich(line, model)
        sr, wav = _speak(line, engine_choice=engine, language=lang)
        p = _save(sr, wav, "batch")
        if p:
            paths.append(p)
        log.append(f"✓ {i + 1}. {line[:60]}")
        yield "\n".join(log), paths
    yield "\n".join(log) + "\n\nГотово / Done.", paths


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def _speaker_blocks():
    with gr.Row():
        num = gr.Slider(2, MAX_SPK, value=2, step=1, label=T("num_speakers"))
        refresh = gr.Button(T("refresh_voices"), size="sm", scale=0)
    choices = [OWN_FILE] + scan_voices()
    blocks, audios, texts, pres = [], [], [], []
    for i in range(MAX_SPK):
        with gr.Group(visible=(i < 2), elem_classes="spk-block") as bl:
            gr.Markdown(f"**Speaker {i}**")
            pre = gr.Dropdown(choices, value=OWN_FILE, label=T("voice_preset"))
            au = gr.Audio(label=T("ref_voice"), type="filepath", sources=["upload", "microphone"])
            tx = gr.Textbox(label=T("ref_text"), lines=1, placeholder=T("ph_clone_tr"))
            pre.change(cb_preset, [pre], [au, tx])
        blocks.append(bl)
        audios.append(au)
        texts.append(tx)
        pres.append(pre)
    num.change(lambda n: [gr.update(visible=(i < n)) for i in range(MAX_SPK)], [num], blocks)
    refresh.click(lambda: [gr.update(choices=[OWN_FILE] + scan_voices()) for _ in range(MAX_SPK)], None, pres)
    return num, audios, texts


LANGUAGE_EXAMPLES = {
    "auto": "Привет! Это современный синтез речи нового поколения с естественными интонациями и живым звучанием.",
    "Russian": "Привет! Это современная студия синтеза речи. Вы можете свободно озвучивать книги, клонировать голоса и создавать новые тембры.",
    "English": "Hello! This is a state-of-the-art speech synthesis studio delivering crystal-clear voice and natural prosody.",
    "Chinese": "你好！这是新一代高品质语音合成系统，能够生成自然流畅、情感丰富的语音。",
    "Cantonese": "你好！呢個係全新嘅高品質語音合成系統，聲音自然流暢，情感豐富。",
    "Japanese": "こんにちは！これは最新の高品質な音声合成モデルで、自然なイントネーションで滑らかに発声します。",
    "Korean": "안녕하세요! 자연스러운 억양과 생생한 음성을 제공하는 최신 음성 합성 시스템입니다.",
    "Spanish": "¡Hola! Este es un sistema avanzado de síntesis de voz que ofrece una entonación natural y expresiva.",
    "French": "Bonjour ! Ceci est un système de synthèse vocale de pointe offrant une diction naturelle et fluide.",
    "Arabic": "مرحبًا! هذا نموذج متطور لتوليد الكلام بنبرة طبيعية وصوت فائق الوضوح.",
    "Turkish": "Merhaba! Bu, doğal tonlama ve son derece akıcı ses kalitesi sunan yeni nesil bir ses sentezi sistemidir.",
    "Indonesian": "Halo! Ini adalah sistem sintesis suara generasi baru dengan intonasi alami dan kualitas audio yang jernih.",
    "Portuguese": "Olá! Este é um sistema avançado de síntese de voz com entonação natural e excelente clareza acústica.",
    "Italian": "Ciao! Questo è un modello avanzato di sintesi vocale che offre un'intonazione naturale e un suono autentico.",
    "Dutch": "Hallo! Dit is een geavanceerd spraaksynthesesysteem met natuurlijke intonatie en kristalhelder geluid.",
    "Vietnamese": "Xin chào! Đây là hệ thống tổng hợp giọng nói thế hệ mới với ngữ điệu tự nhiên và âm thanh sống động.",
    "German": "Hallo! Dies ist ein modernes Sprachsynthesesystem mit natürlicher Intonation und hervorragender Sprachqualität.",
    "Ukrainian": "Привіт! Це сучасна система синтезу мовлення з природною інтонацією та чистим звучанням.",
    "Thai": "สวัสดีครับ! นี่คือระบบสังเคราะห์เสียงพูดคุณภาพสูงที่ให้เสียงเป็นธรรมชาติและคมชัด",
    "Polish": "Cześć! To nowoczesny system syntezy mowy zapewniający naturalną intonację i wysoką jakość głosu.",
    "Romanian": "Bună ziua! Acesta este un sistem avansat de sinteză vocală cu intonație naturală și claritate excelentă.",
    "Greek": "Γειά σας! Αυτό είναι ένα σύστημα σύνθεσης ομιλίας νέας γενιάς με φυσική προσωδία και καθαρό ήχο.",
    "Czech": "Ahoj! Toto je pokročilý systém syntézy řeči s přirozenou intonací a čistým zvukovým podáním.",
    "Finnish": "Hei! Tämä on edistyksellinen puhesynteesijärjestelmä, joka tuottaa luonnollisen ja selkeän äänen.",
    "Hindi": "नमस्ते! यह एक उन्नत वाक् संश्लेषण प्रणाली है जो प्राकृतिक स्वर और स्पष्ट आवाज़ प्रदान करती है।",
    "ZH_Anhui": "您好！这是安徽方言自然语音合成示例，发音地道流畅。",
    "ZH_Fujian": "汝好！这是福建方言自然语音合成示例，发音纯正地道。",
    "ZH_Gansu": "你好！这是甘肃方言语音合成示例，声音自然生动。",
    "ZH_Guizhou": "你好！这是贵州方言自然语音合成示例，音色生动亲切。",
    "ZH_Hebei": "您好！这是河北方言自然语音合成示例，发音清晰地道。",
    "ZH_Henan": "恁好！这是河南方言语音合成示例，中不中？",
    "ZH_Hubei": "你好！这是湖北方言自然语音合成示例，声音纯正自然。",
    "ZH_Hunan": "你好！这是湖南方言语音合成示例，韵味十足。",
    "ZH_Jiangxi": "你好！这是江西方言自然语音合成示例，语调自然地道。",
    "ZH_Liaoning": "你好啊！这是辽宁东北话语音合成示例，老铁听听怎么样！",
    "ZH_Minnan": "汝好！这是闽南语语音合成示例，乡音地道亲切。",
    "ZH_Ningxia": "你好！这是宁夏方言自然语音合成示例，发音纯正自然。",
    "ZH_Shaanxi": "你好！这是陕西话语音合成示例，老陕的乡音地道美！",
    "ZH_Shandong": "您好！这是山东方言语音合成示例，腔调地道醇厚。",
    "ZH_Shanghai": "侬好！这是上海话自然语音合成示例，吴侬软语，声音动听。",
    "ZH_Shanxi": "你好！这是山西方言自然语音合成示例，音韵地道自然。",
    "ZH_Sichuan": "你好哇！这是四川话语音合成示例，巴适得板！",
    "ZH_Tianjin": "您好！这是天津话自然语音合成示例，倍儿地道！",
    "ZH_Wenzhou": "你好！这是温州方言自然语音合成示例，发音地道自然。",
    "ZH_Wu": "侬好！这是吴语方言自然语音合成示例，发音温润流畅。",
    "ZH_Yunnan": "你好！这是云南话自然语音合成示例，声音自然好听。"
}


def cb_load_example(lang="auto"):
    if not lang or lang not in LANGUAGE_EXAMPLES:
        lang = "auto"
    return LANGUAGE_EXAMPLES.get(lang, LANGUAGE_EXAMPLES["auto"])


def _build_tts_panel(engine_name, model_dd=None, lang_dd=None, gui_cfg=None, prefix="t"):
    gui_cfg = gui_cfg or {}
    is_higgs = "Higgs" in engine_name
    DEFAULT_PRESET = "(По умолчанию / Default)"
    with gr.Row():
        with gr.Column():
            t_text = gr.Textbox(label=T("text"), placeholder=T("ph_text"), lines=4, value=gui_cfg.get(f"{prefix}_text", gui_cfg.get("t_text", "")))
            with gr.Row():
                t_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
                t_example_btn = gr.Button(T("example_btn"), size="sm", variant="secondary")
            with gr.Row():
                t_preset = gr.Dropdown([DEFAULT_PRESET] + scan_voices(), value=gui_cfg.get(f"{prefix}_preset", DEFAULT_PRESET), label=T("voice_preset"), scale=2)
                t_refresh = gr.Button(T("refresh"), size="sm", scale=0)
            with gr.Accordion(T("advanced"), open=False):
                t_temp = gr.Slider(0.0, 1.5, gui_cfg.get("t_temp", 1.0), step=0.05, label="Temperature")
                t_top_p = gr.Slider(0.1, 1.0, gui_cfg.get("t_top_p", 0.95), step=0.01, label="Top-p")
                t_top_k = gr.Slider(0, 1026, gui_cfg.get("t_top_k", 50), step=1, label="Top-k (0=off)")
                t_max = gr.Slider(64, 4096, gui_cfg.get("t_max", 2048), step=64, label=T("max_tokens"))
                t_seed = gr.Number(gui_cfg.get("t_seed", -1), label=T("seed"), precision=0)
            t_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("t_auto", False), visible=is_higgs)
            t_btn = gr.Button(T("generate"), variant="primary", size="lg")
            t_stop = gr.Button(T("stop"), variant="stop")
        t_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
    gr.Examples(TTS_EXAMPLES, inputs=[t_text], label=T("examples"))
    t_upload.upload(load_text_file, [t_upload], [t_text])
    t_refresh.click(lambda: gr.update(choices=[DEFAULT_PRESET] + scan_voices()), None, [t_preset])
    eng_state = gr.State(engine_name)
    m_input = model_dd if model_dd is not None else gr.State(dr.DEFAULT_MODEL)
    l_input = lang_dd if lang_dd is not None else gr.State("auto")
    t_example_btn.click(cb_load_example, [l_input], [t_text])
    t_btn.click(save_tts_settings, [t_text, t_temp, t_top_p, t_top_k, t_max, t_seed, eng_state, l_input, t_preset], None)
    ev_tts = t_btn.click(cb_tts, [t_text, m_input, t_auto, t_temp, t_top_p, t_top_k, t_max, t_seed, eng_state, l_input, t_preset],
                         [t_out, t_text])
    t_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_tts])


def _build_expr_panel(model_dd=None, gui_cfg=None):
    gui_cfg = gui_cfg or {}
    e_text = gr.Textbox(label=T("text"), placeholder=T("ph_text"), lines=5, value=gui_cfg.get("e_text", ""))
    with gr.Row():
        e_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
        e_example_btn = gr.Button(T("example_btn"), size="sm", variant="secondary")
    e_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("e_auto", False))
    for cat, clabel in (("emotion", T("cat_emotion")), ("prosody", T("cat_prosody")),
                        ("style", T("cat_style")), ("sfx", T("cat_sfx"))):
        gr.Markdown(f"**{clabel}**")
        with gr.Row():
            for val in sorted(dr.WHITELIST[cat]):
                gr.Button(f"{TAG_RU.get(val, val)}\n{val}", size="sm", elem_classes=["tagbtn"]).click(
                    lambda t, c=cat, v=val: (t or "") + f"<|{c}:{v}|>", [e_text], [e_text])
    with gr.Row():
        e_enrich = gr.Button(T("enrich"), variant="secondary")
        e_btn = gr.Button(T("generate"), variant="primary")
    e_stop = gr.Button(T("stop"), variant="stop")
    e_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
    gr.Examples(EXPR_EXAMPLES, inputs=[e_text], label=T("examples"))
    with gr.Accordion(T("tags_help"), open=True):
        gr.Markdown(TAGS_LEGEND_MD)
    e_upload.upload(load_text_file, [e_upload], [e_text])
    e_example_btn.click(lambda: "<|emotion:elation|>Привет! <|prosody:cheerful|>Это пример эмоционального текста со звуковыми эффектами! <|sfx:laughter|>ха-ха!", None, [e_text])
    eng_state = gr.State(ENGINE_HIGGS)
    lang_state = gr.State("auto")
    m_input = model_dd if model_dd is not None else gr.State(dr.DEFAULT_MODEL)
    e_enrich.click(save_expr_settings, [e_text, e_auto], None)
    e_enrich.click(cb_enrich, [e_text, m_input], [e_text])
    e_btn.click(save_expr_settings, [e_text, e_auto], None)
    ev_expr = e_btn.click(cb_expr, [e_text, m_input, e_auto, eng_state, lang_state], [e_out, e_text])
    e_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_expr])


def _build_clone_panel(engine_name, model_dd=None, lang_dd=None, gui_cfg=None, prefix="c"):
    gui_cfg = gui_cfg or {}
    is_higgs = "Higgs" in engine_name
    with gr.Row():
        with gr.Column():
            c_text = gr.Textbox(label=T("text"), placeholder=T("ph_clone"), lines=3, value=gui_cfg.get(f"{prefix}_text", gui_cfg.get("c_text", "")))
            with gr.Row():
                c_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
                c_example_btn = gr.Button(T("example_btn"), size="sm", variant="secondary")
            c_preset = gr.Dropdown([OWN_FILE] + scan_voices(), value=gui_cfg.get("c_preset", OWN_FILE), label=T("voice_preset"))
            c_refresh = gr.Button(T("refresh"), size="sm")
            c_ref = gr.Audio(label=T("ref_voice"), type="filepath", sources=["upload", "microphone"])
            c_ref_text = gr.Textbox(label=T("ref_text"), lines=2, placeholder=T("ph_clone_tr"))
            c_tr_btn = gr.Button(T("transcribe_btn"), size="sm")
            c_temp = gr.Slider(0.0, 1.5, gui_cfg.get("c_temp", 1.0), step=0.05, label="Temperature")
            c_top_p = gr.Slider(0.1, 1.0, gui_cfg.get("c_top_p", 0.95), step=0.01, label="Top-p")
            c_seed = gr.Number(gui_cfg.get("c_seed", -1), label=T("seed"), precision=0)
            c_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("c_auto", False), visible=is_higgs)
            c_btn = gr.Button(T("generate"), variant="primary", size="lg")
            c_stop = gr.Button(T("stop"), variant="stop")
        c_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
    with gr.Accordion(T("cloud_title"), open=False):
        cl_status = gr.Textbox(label=T("cloud_status"), interactive=False)
        with gr.Row():
            cl_load = gr.Button(T("load_list"), size="sm")
            cl_all = gr.Button(T("download_all"), size="sm")
        cl_voices = gr.CheckboxGroup(choices=[], label=T("cloud_voices"))
        cl_dl = gr.Button(T("download_sel"), variant="primary", size="sm")
    c_preset.change(cb_preset, [c_preset], [c_ref, c_ref_text])
    c_tr_btn.click(transcribe, [c_ref], [c_ref_text])
    c_refresh.click(lambda: gr.update(choices=[OWN_FILE] + scan_voices()), None, [c_preset])
    cl_load.click(cb_load_cloud, None, [cl_status, cl_voices])
    cl_all.click(cb_download_all_cloud, None, [cl_status, c_preset])
    cl_dl.click(cb_download_voices, [cl_voices], [cl_status, c_preset])
    c_upload.upload(load_text_file, [c_upload], [c_text])
    eng_state = gr.State(engine_name)
    m_input = model_dd if model_dd is not None else gr.State(dr.DEFAULT_MODEL)
    l_input = lang_dd if lang_dd is not None else gr.State("auto")
    c_example_btn.click(cb_load_example, [l_input], [c_text])
    c_btn.click(save_clone_settings, [c_text, c_preset, c_auto, c_temp, c_top_p, c_seed, eng_state, l_input], None)
    ev_clone = c_btn.click(cb_clone, [c_text, m_input, c_auto, c_ref, c_ref_text, c_preset, c_temp, c_top_p, c_seed, eng_state, l_input],
                           [c_out, c_text])
    c_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_clone])


def _build_voice_design_panel(lang_dd=None):
    gr.Markdown(T("design_hint"))
    with gr.Row():
        with gr.Column():
            vd_inst = gr.Textbox(label=T("design_inst"), placeholder=T("ph_design_inst"), lines=3,
                                 value="A warm, confident male voice in his 30s speaking Russian at a steady pace.")
            vd_text = gr.Textbox(label=T("text"), placeholder=T("ph_text"), lines=4,
                                 value="Здравствуйте! Это синтез речи с уникальным тембром, созданным по текстовому описанию.")
            vd_example_btn = gr.Button(T("example_btn"), size="sm", variant="secondary")
            with gr.Accordion(T("advanced"), open=False):
                vd_lang = lang_dd if lang_dd is not None else gr.Dropdown(fe.LANGUAGE_CHOICES, value="auto", label=T("lang_label"))
                vd_temp = gr.Slider(0.1, 1.5, 0.7, step=0.05, label="Text Temperature")
                vd_top_p = gr.Slider(0.1, 1.0, 0.8, step=0.01, label="Text Top-p")
                vd_top_k = gr.Slider(0, 100, 20, step=1, label="Text Top-k")
                vd_rep = gr.Slider(1.0, 2.0, 1.0, step=0.05, label="Repetition Penalty")
                vd_steps = gr.Slider(1, 30, 10, step=1, label="DiT Flow Steps")
                vd_cfg = gr.Slider(0.0, 3.0, 1.2, step=0.1, label="CFG")
                vd_seed = gr.Number(-1, label=T("seed"), precision=0)
                vd_max_sec = gr.Slider(10.0, 120.0, 64.0, step=5.0, label="Max Audio Sec")
                vd_tn = gr.Checkbox(label="Text Normalization (TN)", value=True)
                vd_split = gr.Checkbox(label="Split Sentences", value=True)
                vd_cross = gr.Slider(0.0, 200.0, 50.0, step=10.0, label="Cross-fade ms")
            vd_btn = gr.Button(T("generate"), variant="primary", size="lg")
            vd_stop = gr.Button(T("stop"), variant="stop")
        with gr.Column():
            vd_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
            vd_plan = gr.Textbox(label=T("voice_plan_label"), lines=6, interactive=False)
    vd_example_btn.click(cb_load_example, [vd_lang], [vd_text])
    ev_vd = vd_btn.click(
        cb_voice_design,
        [vd_inst, vd_text, vd_lang, vd_temp, vd_top_p, vd_top_k, vd_rep, vd_steps, vd_cfg, vd_seed, vd_max_sec, vd_tn, vd_split, vd_cross],
        [vd_out, vd_plan]
    )
    vd_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_vd])


def _build_speech_edit_panel():
    gr.Markdown(T("edit_hint"))
    with gr.Row():
        with gr.Column():
            ed_audio = gr.Audio(label=T("edit_audio_label"), type="filepath", sources=["upload", "microphone"])
            ed_mode = gr.Radio([("Семантическое (слова)", "semantic"), ("Акустическое (скорость/тон/громкость)", "acoustic")], value="semantic", label=T("edit_mode_label"))

            with gr.Group(visible=True) as ed_sem_group:
                ed_sem_inst = gr.Textbox(label=T("sem_inst_label"), placeholder=T("ph_sem_inst"),
                                         value="Replace 'dogs' with 'cats'.")

            with gr.Group(visible=False) as ed_ac_group:
                ed_ac_param = gr.Dropdown(["speed", "pitch", "volume"], value="speed", label=T("ac_mode_label"))
                ed_ac_val = gr.Slider(-6.0, 6.0, 1.2, step=0.1, label=T("ac_val_label"))
                ed_ac_custom = gr.Textbox(label=T("custom_inst_label"), placeholder="adjust the speed to 1.2x")

            with gr.Accordion(T("advanced"), open=False):
                ed_steps = gr.Slider(1, 30, 10, step=1, label="DiT Flow Steps")
                ed_cfg = gr.Slider(0.0, 3.0, 1.2, step=0.1, label="CFG")
                ed_th = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="Stop Threshold")
                ed_seed = gr.Number(-1, label=T("seed"), precision=0)
                ed_max_sec = gr.Slider(10.0, 120.0, 64.0, step=5.0, label="Max Audio Sec")

            ed_btn = gr.Button(T("generate"), variant="primary", size="lg")
            ed_stop = gr.Button(T("stop"), variant="stop")
        with gr.Column():
            ed_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
            ed_text_out = gr.Textbox(label=T("edited_text_label"), lines=4, interactive=False)

    def on_edit_mode_change(m):
        is_sem = (m == "semantic")
        return gr.update(visible=is_sem), gr.update(visible=not is_sem)

    ed_mode.change(on_edit_mode_change, [ed_mode], [ed_sem_group, ed_ac_group])

    def run_edit(mode, audio, sem_i, ac_p, ac_v, ac_c, steps, cfg, th, seed, max_s):
        if not audio:
            return None, "Загрузите аудиофайл"
        if mode == "semantic":
            return cb_semantic_edit(audio, sem_i, steps, cfg, th, seed, max_s)
        else:
            res_audio = cb_acoustic_edit(audio, ac_p, ac_v, ac_c, steps, cfg, th, seed, max_s)
            return res_audio, "Акустическое редактирование завершено"

    ev_ed = ed_btn.click(run_edit, [ed_mode, ed_audio, ed_sem_inst, ed_ac_param, ed_ac_val, ed_ac_custom, ed_steps, ed_cfg, ed_th, ed_seed, ed_max_sec], [ed_out, ed_text_out])
    ed_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_ed])


def _build_long_clone_panel(engine_name, model_dd=None, lang_dd=None, gui_cfg=None, llm_cfg=None, prefix="lc"):
    gui_cfg = gui_cfg or {}
    llm_cfg = llm_cfg or {}
    is_higgs = "Higgs" in engine_name
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    lc_custom_num = gr.Checkbox(label=T("lc_custom_num"), value=gui_cfg.get("lc_custom_num", False), scale=1)
                    lc_num_input = gr.Textbox(value=gui_cfg.get("lc_num_input", "01"), scale=0, min_width=100, show_label=False)
            lc_btn_top = gr.Button(T("generate"), variant="primary", size="lg")
            lc_text = gr.Textbox(label=T("text"), placeholder=T("ph_clone"), lines=5, value=gui_cfg.get(f"{prefix}_text", gui_cfg.get("lc_text", "")))
            with gr.Row():
                lc_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
                lc_example_btn = gr.Button(T("example_btn"), size="sm", variant="secondary")
            lc_processed = gr.Textbox(label=T("processed_text_label"), lines=5, interactive=False)
            lc_preset = gr.Dropdown([OWN_FILE] + scan_voices(), value=gui_cfg.get("lc_preset", OWN_FILE), label=T("voice_preset"))
            lc_refresh = gr.Button(T("refresh"), size="sm")
            lc_ref = gr.Audio(label=T("ref_voice"), type="filepath", sources=["upload", "microphone"])
            lc_ref_text = gr.Textbox(label=T("ref_text"), lines=2, placeholder=T("ph_clone_tr"))
            lc_tr_btn = gr.Button(T("transcribe_btn"), size="sm")
            lc_max_chars = gr.Slider(50, 500, gui_cfg.get("lc_max_chars", 150), step=10, label=T("max_chars_label"), info=T("max_chars_info"))
            lc_gap = gr.Slider(0.0, 2.0, gui_cfg.get("lc_gap", 0.3), step=0.1, label=T("gap_label"))
            lc_merge = gr.Checkbox(label=T("merge_label"), value=gui_cfg.get("lc_merge", True))
            with gr.Accordion(T("advanced"), open=False):
                lc_temp = gr.Slider(0.0, 1.5, gui_cfg.get("lc_temp", 1.0), step=0.05, label="Temperature")
                lc_top_p = gr.Slider(0.1, 1.0, gui_cfg.get("lc_top_p", 0.95), step=0.01, label="Top-p")
                lc_seed = gr.Number(gui_cfg.get("lc_seed", -1), label=T("seed"), precision=0)
            lc_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("lc_auto", False), visible=is_higgs)

            with gr.Accordion(T("llm_settings_title"), open=False, visible=is_higgs):
                lc_api_url = gr.Textbox(label=T("api_url_label"), value=llm_cfg.get("api_url", "http://localhost:1234/v1"))
                lc_api_key = gr.Textbox(label=T("api_key_label"), value=llm_cfg.get("api_key", ""), type="password")
                lc_api_model = gr.Textbox(label=T("api_model_label"), value=llm_cfg.get("api_model", "gemma-2-12b-it"))
                lc_system_prompt = gr.Textbox(label=T("system_prompt_label"), value=llm_cfg.get("system_prompt", ""), lines=4)
                lc_save_btn = gr.Button(T("save_settings_btn"), variant="secondary")
                lc_test_btn = gr.Button(T("test_connection_btn"), variant="secondary")
                lc_save_status = gr.Markdown()

                lc_save_btn.click(
                    save_llm_config,
                    [lc_api_url, lc_api_key, lc_api_model, lc_system_prompt],
                    [lc_save_status]
                )
                lc_test_btn.click(
                    cb_test_llm_connection,
                    [lc_api_url, lc_api_key, lc_api_model],
                    [lc_save_status]
                )

            lc_btn = gr.Button(T("generate"), variant="primary", size="lg")
            lc_stop = gr.Button(T("stop"), variant="stop")
        with gr.Column():
            lc_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
            lc_log = gr.Textbox(label=T("long_clone_log"), lines=12)
            lc_files = gr.Files(label=T("result"), elem_classes=["scrollable-files"])
    lc_preset.change(cb_preset, [lc_preset], [lc_ref, lc_ref_text])
    lc_tr_btn.click(transcribe, [lc_ref], [lc_ref_text])
    lc_refresh.click(lambda: gr.update(choices=[OWN_FILE] + scan_voices()), None, [lc_preset])
    lc_upload.upload(load_text_file, [lc_upload], [lc_text])
    eng_state = gr.State(engine_name)
    m_input = model_dd if model_dd is not None else gr.State(dr.DEFAULT_MODEL)
    l_input = lang_dd if lang_dd is not None else gr.State("auto")
    lc_example_btn.click(cb_load_example, [l_input], [lc_text])
    for btn in [lc_btn, lc_btn_top]:
        btn.click(save_long_clone_settings, [lc_text, lc_preset, lc_max_chars, lc_gap, lc_merge, lc_temp, lc_top_p, lc_seed, lc_auto, lc_custom_num, lc_num_input, eng_state, l_input], None)
    ev_long_clone = lc_btn.click(cb_long_clone, [lc_text, m_input, lc_auto, lc_ref, lc_ref_text, lc_preset, lc_temp, lc_top_p, lc_seed, lc_max_chars, lc_gap, lc_merge, lc_api_url, lc_api_key, lc_api_model, lc_system_prompt, lc_custom_num, lc_num_input, eng_state, l_input],
                                 [lc_log, lc_out, lc_files, lc_processed, lc_num_input])
    ev_long_clone_top = lc_btn_top.click(cb_long_clone, [lc_text, m_input, lc_auto, lc_ref, lc_ref_text, lc_preset, lc_temp, lc_top_p, lc_seed, lc_max_chars, lc_gap, lc_merge, lc_api_url, lc_api_key, lc_api_model, lc_system_prompt, lc_custom_num, lc_num_input, eng_state, l_input],
                                         [lc_log, lc_out, lc_files, lc_processed, lc_num_input])
    lc_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_long_clone, ev_long_clone_top])


def _build_podcast_panel(engine_name, model_dd=None, lang_dd=None):
    gr.Markdown(T("pod_hint"))
    gr.Markdown(T("pod_format"))
    p_num, p_audios, p_texts = _speaker_blocks()
    p_topic = gr.Textbox(label=T("topic"), placeholder=T("ph_topic"), lines=2)
    gr.Examples(POD_TOPICS, inputs=[p_topic], label=T("examples"))
    p_model = model_dd if model_dd is not None else gr.Dropdown(MODEL_CHOICES, value=dr.DEFAULT_MODEL, label=T("director_model"))
    p_script_btn = gr.Button(T("make_script"), variant="secondary")
    p_script = gr.Textbox(label=T("script"), placeholder=T("ph_script"), lines=9)
    p_btn = gr.Button(T("synth"), variant="primary", size="lg")
    p_stop = gr.Button(T("stop"), variant="stop")
    p_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
    p_script_btn.click(cb_podcast_script, [p_topic, p_num, p_model], [p_script])
    eng_state = gr.State(engine_name)
    l_input = lang_dd if lang_dd is not None else gr.State("auto")
    ev_pod = p_btn.click(cb_multi_synth, [p_script] + p_audios + p_texts + [eng_state, l_input], p_out)
    p_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_pod])


def _build_audiobook_panel(engine_name, model_dd=None, lang_dd=None, gui_cfg=None):
    gui_cfg = gui_cfg or {}
    gr.Markdown(T("book_hint"))
    gr.Markdown(T("book_format"))
    b_num, b_audios, b_texts = _speaker_blocks()
    b_text = gr.Textbox(label=T("book_text"), placeholder=T("ph_book"), lines=6, value=gui_cfg.get("b_text", ""))
    b_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
    gr.Examples(BOOK_EXAMPLES, inputs=[b_text], label=T("examples"))
    b_model = model_dd if model_dd is not None else gr.Dropdown(MODEL_CHOICES, value=gui_cfg.get("model_dd", dr.DEFAULT_MODEL), label=T("director_model"))
    b_markup = gr.Button(T("markup"), variant="secondary")
    b_script = gr.Textbox(label=T("script"), placeholder=T("ph_script"), lines=9)
    b_btn = gr.Button(T("synth"), variant="primary", size="lg")
    b_stop = gr.Button(T("stop"), variant="stop")
    b_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
    b_upload.upload(load_text_file, [b_upload], [b_text])
    b_markup.click(save_book_settings, [b_text], None)
    b_markup.click(cb_book_markup, [b_text, b_num, b_model], [b_script])
    b_btn.click(save_book_settings, [b_text], None)
    eng_state = gr.State(engine_name)
    l_input = lang_dd if lang_dd is not None else gr.State("auto")
    ev_book = b_btn.click(cb_multi_synth, [b_script] + b_audios + b_texts + [eng_state, l_input], b_out)
    b_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_book])


def _build_batch_panel(engine_name, model_dd=None, lang_dd=None, gui_cfg=None, prefix="bt"):
    gui_cfg = gui_cfg or {}
    is_higgs = "Higgs" in engine_name
    bt_text = gr.Textbox(label=T("batch_text"), placeholder=T("ph_batch"), lines=6, value=gui_cfg.get(f"{prefix}_text", gui_cfg.get("bt_text", "")))
    bt_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
    bt_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("bt_auto", False), visible=is_higgs)
    bt_btn = gr.Button(T("generate"), variant="primary", size="lg")
    bt_stop = gr.Button(T("stop"), variant="stop")
    bt_log = gr.Textbox(label=T("log"), lines=8)
    bt_files = gr.Files(label=T("result"), elem_classes=["scrollable-files"])
    bt_upload.upload(load_text_file, [bt_upload], [bt_text])
    bt_btn.click(save_batch_settings, [bt_text, bt_auto], None)
    eng_state = gr.State(engine_name)
    m_input = model_dd if model_dd is not None else gr.State(dr.DEFAULT_MODEL)
    l_input = lang_dd if lang_dd is not None else gr.State("auto")
    ev_batch = bt_btn.click(cb_batch, [bt_text, m_input, bt_auto, eng_state, l_input], [bt_log, bt_files])
    bt_stop.click(lambda: (eng.request_cancel(), fe.request_cancel()), None, None, queue=False, cancels=[ev_batch])


def build():
    llm_cfg = load_llm_config()
    gui_cfg = load_gui_config()
    with gr.Blocks(title=APP_NAME, css=CSS, head=HEAD_SCRIPT, js=DARK_JS) as demo:
        gr.HTML(T("brand_header_html"))

        with gr.Row():
            cpu_cb = gr.Checkbox(label=T("cpu_only_label"), value=gui_cfg.get("cpu_only", False), scale=1)
            keep_cb = gr.Checkbox(label=T("keep_vram_label"), value=gui_cfg.get("keep_vram", False), scale=1)
        with gr.Row():
            fmt_dd = gr.Radio(["mp3", "wav", "flac", "ogg"], value=gui_cfg.get("out_format", "mp3"), label=T("out_format"), elem_classes=["fmt-radio-row"], scale=1)
            quant_dd = gr.Dropdown([("bf16 — высокое качество (рекомендуется)", "bf16"),
                                    ("8-bit / INT8 — экономия памяти", "8bit"),
                                    ("4-bit — минимум VRAM (только Higgs)", "4bit"),
                                    ("fp32 — полная точность", "fp32")],
                                   value=gui_cfg.get("precision", "bf16"), label=T("quant"), scale=1)

        def on_cpu_change(val):
            update_gui_config("cpu_only", val)
            eng.set_cpu_mode(val)
            fe.set_cpu_mode(val)
            dr.set_cpu_mode(val)
            eng.unload_tts(force=True)
            fe.unload_tts(force=True)
            dr.unload_llm(force=True)

        cpu_cb.change(on_cpu_change, [cpu_cb], None)

        def on_keep_change(val):
            update_gui_config("keep_vram", val)
            if not val:
                eng.unload_tts(force=True)
                fe.unload_tts(force=True)
                dr.unload_llm(force=True)

        keep_cb.change(on_keep_change, [keep_cb], None)

        def on_quant_change(p):
            update_gui_config("precision", p)
            eng.set_precision(p)
            fe.set_precision("int8" if p == "8bit" else p)

        quant_dd.change(on_quant_change, [quant_dd], None)

        def on_format_change(f):
            set_out_format(f)
            update_gui_config("out_format", f)

        fmt_dd.change(on_format_change, [fmt_dd], None)

        # Главные вкладки переключения движков (Higgs Audio v3.1 / FireRedTTS3)
        with gr.Tabs(elem_classes=["engine-main-tabs"], selected=gui_cfg.get("active_engine_index", 0)) as engine_tabs:
            # =========================================================================
            # Вкладка 1: Higgs Audio v3.1 (100+ языков, Эмоции & Режиссёр)
            # =========================================================================
            with gr.Tab(T("engine_higgs_tab"), id=0):
                with gr.Row(elem_classes=["engine-dropdown-row"]):
                    h_model_dd = gr.Dropdown(MODEL_CHOICES, value=gui_cfg.get("model_dd", dr.DEFAULT_MODEL), label=T("director_model"), scale=2)
                h_model_dd.change(lambda x: update_gui_config("model_dd", x), [h_model_dd], None)

                with gr.Tabs(elem_classes=["sub-tabs"], selected=gui_cfg.get("h_subtab", 0)) as h_subtabs:
                    with gr.Tab(T("tab_tts"), id=0):
                        _build_tts_panel(ENGINE_HIGGS, model_dd=h_model_dd, gui_cfg=gui_cfg, prefix="t_h")
                    with gr.Tab(T("tab_expr"), id=1):
                        _build_expr_panel(model_dd=h_model_dd, gui_cfg=gui_cfg)
                    with gr.Tab(T("tab_clone"), id=2):
                        _build_clone_panel(ENGINE_HIGGS, model_dd=h_model_dd, gui_cfg=gui_cfg, prefix="c_h")
                    with gr.Tab(T("tab_long_clone"), id=3):
                        _build_long_clone_panel(ENGINE_HIGGS, model_dd=h_model_dd, gui_cfg=gui_cfg, llm_cfg=llm_cfg, prefix="lc_h")
                    with gr.Tab(T("tab_pod"), id=4):
                        _build_podcast_panel(ENGINE_HIGGS, model_dd=h_model_dd)
                    with gr.Tab(T("tab_book"), id=5):
                        _build_audiobook_panel(ENGINE_HIGGS, model_dd=h_model_dd, gui_cfg=gui_cfg)
                    with gr.Tab(T("tab_batch"), id=6):
                        _build_batch_panel(ENGINE_HIGGS, model_dd=h_model_dd, gui_cfg=gui_cfg, prefix="bt_h")
                h_subtabs.select(lambda evt: update_gui_config("h_subtab", evt.index), None, None)

            # =========================================================================
            # Вкладка 2: FireRedTTS3 (24 языка, Voice Design, Редактирование речи)
            # =========================================================================
            with gr.Tab(T("engine_firered_tab"), id=1):
                with gr.Row(elem_classes=["engine-dropdown-row"]):
                    fr_lang_dd = gr.Dropdown(fe.LANGUAGE_CHOICES, value=gui_cfg.get("language_choice", "auto"), label=T("lang_label"), scale=2)
                fr_lang_dd.change(lambda x: update_gui_config("language_choice", x), [fr_lang_dd], None)

                with gr.Tabs(elem_classes=["sub-tabs"], selected=gui_cfg.get("fr_subtab", 0)) as fr_subtabs:
                    with gr.Tab(T("tab_tts"), id=0):
                        _build_tts_panel(ENGINE_FIRERED, lang_dd=fr_lang_dd, gui_cfg=gui_cfg, prefix="t_fr")
                    with gr.Tab(T("tab_clone"), id=1):
                        _build_clone_panel(ENGINE_FIRERED, lang_dd=fr_lang_dd, gui_cfg=gui_cfg, prefix="c_fr")
                    with gr.Tab(T("tab_design"), id=2):
                        _build_voice_design_panel(lang_dd=fr_lang_dd)
                    with gr.Tab(T("tab_edit"), id=3):
                        _build_speech_edit_panel()
                    with gr.Tab(T("tab_long_clone"), id=4):
                        _build_long_clone_panel(ENGINE_FIRERED, lang_dd=fr_lang_dd, gui_cfg=gui_cfg, llm_cfg=llm_cfg, prefix="lc_fr")
                    with gr.Tab(T("tab_pod"), id=5):
                        _build_podcast_panel(ENGINE_FIRERED, lang_dd=fr_lang_dd)
                    with gr.Tab(T("tab_book"), id=6):
                        _build_audiobook_panel(ENGINE_FIRERED, lang_dd=fr_lang_dd, gui_cfg=gui_cfg)
                    with gr.Tab(T("tab_batch"), id=7):
                        _build_batch_panel(ENGINE_FIRERED, lang_dd=fr_lang_dd, gui_cfg=gui_cfg, prefix="bt_fr")
                fr_subtabs.select(lambda evt: update_gui_config("fr_subtab", evt.index), None, None)

        def on_engine_tab_select(evt: gr.SelectData):
            update_gui_config("active_engine_index", evt.index)

        engine_tabs.select(on_engine_tab_select, None, None)

    return demo


def prewarm():
    """Прогрев в главном потоке."""
    if eng._MOCK:
        return
    try:
        print("[prewarm] Проверка доступности моделей...", flush=True)
    except Exception as e:
        print(f"[prewarm] {e}", flush=True)


if __name__ == "__main__":
    print(f"[{APP_NAME}] {DEVICE_INFO}")
    cfg = load_gui_config()
    cpu_val = cfg.get("cpu_only", False)
    eng.set_cpu_mode(cpu_val)
    fe.set_cpu_mode(cpu_val)
    dr.set_cpu_mode(cpu_val)
    precision_val = cfg.get("precision", "bf16")
    eng.set_precision(precision_val)
    fe.set_precision("int8" if precision_val == "8bit" else precision_val)
    set_out_format(cfg.get("out_format", "mp3"))
    prewarm()
    demo = build().queue(default_concurrency_limit=1)

    _, local_url, _ = demo.launch(
        server_port=None,
        inbrowser=False,
        prevent_thread_lock=True,
        i18n=I18N, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
        css=CSS, js=DARK_JS, head=HEAD_SCRIPT, show_error=True
    )

    if not eng._MOCK and os.environ.get("NO_AUTO_BROWSER", "").lower() not in ("1", "true", "yes"):
        import webbrowser
        saved_lang = cfg.get("ui_lang", "ru")
        url = f"{local_url}?__lang={saved_lang}&__theme=dark"
        print(f"[UI] Opening browser with theme: {url}", flush=True)
        webbrowser.open(url)

    demo.block_thread()
