"""Higgs Audio Studio — портативная сборка Nerual Dreming + Нейро-Софт.


Higgs Audio v3.1 TTS (100+ языков, клонирование) + AI-режиссёр текста (теги по смыслу) +
мульти-спикерные режимы Подкаст и Аудиокнига. UI RU/EN, тёмная тема.
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


APP_NAME = "Higgs Audio Studio"
DEVICE_INFO = eng.device_info()
MODEL_CHOICES = list(dr.MODELS.keys()) + ["External API (LM Studio / Ollama / OpenAI)"]
MAX_SPK = 4
OWN_FILE = "— свой файл / own file —"


CLOUD_VOICES_REPO = "Slait/russia_voices"
CLOUD_VOICES_BASE = "https://huggingface.co/datasets/Slait/russia_voices/resolve/main"




# ----------------------------------------------------------------------------
# Брендинг
# ----------------------------------------------------------------------------
_FLAG = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg"


_DONATE_RU = """
<div class="donate-popover">
  <p class="donate-intro">Привет! Я Илья (<a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a>), я создаю AI-инструменты, которые работают локально — бесплатно, без облака, без подписок. Ваш донат позволяет фокусироваться на исследовании и создании новых открытых проектов, а не на выживании. Спасибо!</p>
  <div class="donate-sep"></div>
  <div class="donate-row"><a href="https://dalink.to/nerual_dreming" target="_blank">💳 Карта / PayPal (рубли, доллары, евро)</a></div>
  <div class="donate-row"><a href="https://boosty.to/neuro_art" target="_blank">🚀 Ежемесячная подписка на Boosty</a></div>
  <div class="donate-sep"></div>
  <div class="donate-row"><span>BTC</span><code>1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC</code></div>
  <div class="donate-row"><span>ETH</span><code>0xb5db65adf478983186d4897ba92fe2c25c594a0c</code></div>
  <div class="donate-row"><span>USDT TRC20</span><code>TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C</code></div>
</div>
"""
_DONATE_EN = """
<div class="donate-popover">
  <p class="donate-intro">Hi! I'm Ilya (<a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a>), I build AI tools that run locally — for free, without cloud, without subscriptions. Your donation lets me focus on research and new open-source projects instead of just surviving. Thank you!</p>
  <div class="donate-sep"></div>
  <div class="donate-row"><a href="https://dalink.to/nerual_dreming" target="_blank">💳 Card / PayPal (USD, EUR, RUB)</a></div>
  <div class="donate-row"><a href="https://boosty.to/neuro_art" target="_blank">🚀 Monthly subscription on Boosty</a></div>
  <div class="donate-sep"></div>
  <div class="donate-row"><span>BTC</span><code>1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC</code></div>
  <div class="donate-row"><span>ETH</span><code>0xb5db65adf478983186d4897ba92fe2c25c594a0c</code></div>
  <div class="donate-row"><span>USDT TRC20</span><code>TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C</code></div>
</div>
"""




def _brand(subtitle, credits, donate, donate_label):
    return f"""
<div class="brand-header">
  <div class="lang-switcher">
    <a href="?__lang=ru&amp;__theme=dark" class="lang-btn"><img src="{_FLAG}/1f1f7-1f1fa.svg" width="16" height="16"/>RU</a>
    <a href="?__lang=en&amp;__theme=dark" class="lang-btn"><img src="{_FLAG}/1f1ec-1f1e7.svg" width="16" height="16"/>EN</a>
    <details class="donate-wrap"><summary class="lang-btn donate-btn"><img src="{_FLAG}/1fa99.svg" width="16" height="16"/>{donate_label}</summary>{donate}</details>
  </div>
  <div class="brand-box">
    <div class="brand-title">🎙️ {APP_NAME}</div>
    <div class="brand-subtitle">{subtitle}</div>
    <div class="brand-credits">{credits}</div>
    <div class="device-badge">💻 {DEVICE_INFO}</div>
  </div>
</div>
"""




BRAND_HTML_RU = _brand(
    "Higgs Audio v3.1 · 100+ языков · клонирование · AI-режиссёр текста · подкаст и аудиокнига",
    'Собрал <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> — '
    'основатель <a href="https://artgeneration.me" target="_blank">ArtGeneration.me</a>, '
    'техноблогер и нейро-евангелист. Канал '
    '<a href="https://t.me/neuroport" target="_blank">Нейро-Софт</a> — репаки и портативки нейросетей.',
    _DONATE_RU, "Донат",
)
BRAND_HTML_EN = _brand(
    "Higgs Audio v3.1 · 100+ languages · voice cloning · AI text director · podcast & audiobook",
    'Built by <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> — '
    'founder of <a href="https://artgeneration.me" target="_blank">ArtGeneration.me</a>, '
    'tech-blogger and neuro-evangelist. Channel '
    '<a href="https://t.me/neuroport" target="_blank">Нейро-Софт</a> — portable AI builds.',
    _DONATE_EN, "Donate",
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
# i18n (gr.I18n, регистрируется в launch(i18n=))
# ----------------------------------------------------------------------------
_RU = {
    "tab_tts": "🎙️ Озвучка", "tab_expr": "🎭 Экспрессия + Режиссёр", "tab_clone": "🧬 Клонирование",
    "tab_pod": "🎬 Подкаст", "tab_book": "📚 Аудиокнига", "tab_batch": "📦 Пакет",
    "text": "Текст", "ph_text": "Введите текст… можно вставлять теги: <|emotion:elation|>Привет!",
    "generate": "🔊 Озвучить", "stop": "⏹ Стоп", "result": "Результат", "advanced": "Доп. настройки",
    "director_model": "Модель режиссёра (для обогащения / диалогов)",
    "quant": "Квантизация Higgs ⚗️ (экспериментально)",
    "quant_info": "⚗️ Экспериментально: квантизация (4/8-bit) экономит VRAM, но качество звука заметно страдает. Для лучшего качества оставьте bf16.",
    "out_format": "Формат вывода",
    "cat_emotion": "😊 Эмоции (на предложение)", "cat_prosody": "🎵 Просодия", "cat_style": "🎭 Стиль", "cat_sfx": "🔊 Звуки (по месту в тексте)",
    "download_all": "⬇️ Скачать все 700+",
    "enrich": "✨ Обогатить текст", "auto_enrich": "✨ Авто-обогащение промпта режиссёром",
    "upload_txt": "📂 Загрузить текст из файла (.txt)",
    "ref_voice": "Аудио-референс (голос)", "ref_text": "Транскрипт референса (заполнится сам)",
    "ph_clone_tr": "Что произносится в референсе…", "voice_preset": "Пресет голоса",
    "refresh": "🔄 Обновить", "transcribe_btn": "📝 Распознать транскрипт",
    "seed": "Сид (-1 = случайно)", "max_tokens": "Макс. токенов",
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
    "keep_vram_label": "⚡ Не выгружать модели из памяти (ускоряет повторные запуски; для TTS достаточно 6 ГБ+ VRAM)",
}
_EN = {
    "tab_tts": "🎙️ TTS", "tab_expr": "🎭 Expressive + Director", "tab_clone": "🧬 Cloning",
    "tab_pod": "🎬 Podcast", "tab_book": "📚 Audiobook", "tab_batch": "📦 Batch",
    "text": "Text", "ph_text": "Type text… you can insert tags: <|emotion:elation|>Hi!",
    "generate": "🔊 Generate", "stop": "⏹ Stop", "result": "Result", "advanced": "Advanced",
    "director_model": "Director model (enrich / dialogues)",
    "quant": "Higgs quantization ⚗️ (experimental)",
    "quant_info": "⚗️ Experimental: quantization (4/8-bit) saves VRAM but audibly degrades quality. Keep bf16 for best quality.",
    "out_format": "Output format",
    "cat_emotion": "😊 Emotion (per sentence)", "cat_prosody": "🎵 Prosody", "cat_style": "🎭 Style", "cat_sfx": "🔊 Sounds (inline)",
    "download_all": "⬇️ Download all 700+",
    "enrich": "✨ Enrich text", "auto_enrich": "✨ Auto-enrich prompt with director",
    "upload_txt": "📂 Upload text from file (.txt)",
    "ref_voice": "Reference audio (voice)", "ref_text": "Reference transcript (auto-filled)",
    "ph_clone_tr": "What the reference says…", "voice_preset": "Voice preset",
    "refresh": "🔄 Refresh", "transcribe_btn": "📝 Transcribe reference",
    "seed": "Seed (-1 = random)", "max_tokens": "Max tokens",
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
    "keep_vram_label": "⚡ Keep models in VRAM (speeds up subsequent runs; needs only 6 GB+ VRAM for pure TTS)",
}
I18N = gr.I18n(en=_EN, ru=_RU)




def T(key):
    return I18N(key)




HEAD_SCRIPT = """
<script>
(function(){
  var lang;
  try { lang = new URL(window.location).searchParams.get('__lang'); } catch(e) { lang = null; }
  if (!lang) return;  // нет ?__lang= → язык браузера (дефолт Gradio)
  // navigator override (геттером — стабильный API) для встроенных строк Gradio
  try {
    Object.defineProperty(navigator, 'language',  {get: function(){ return lang; }, configurable: true});
    Object.defineProperty(navigator, 'languages', {get: function(){ return [lang]; }, configurable: true});
    document.documentElement.lang = lang;
  } catch(e) {}
  // Главный рычаг: writable-стор локали svelte-i18n. Находим ПО ФОРМЕ (subscribe+set, значение —
  // строка-локаль), НЕ по минифицированному имени (оно меняется между сборками Gradio). set() ретранслирует UI.
  var sp = null;
  function getStore(){
    if (sp) return sp;
    var link = document.querySelector('link[href*="i18n-"]');
    if (!link) return null;
    sp = import(link.href).then(function(m){
      for (var k in m){ var v = m[k];
        try { if (v && typeof v.subscribe === 'function' && typeof v.set === 'function'){
          var c; var u = v.subscribe(function(x){ c = x; }); if (typeof u === 'function') u();
          if (typeof c === 'string' && /^[a-z]{2}(-[A-Za-z]+)?$/.test(c)) return v;
        }} catch(e) {}
      }
      return null;
    }).catch(function(){ return null; });
    return sp;
  }
  function apply(){
    var p = getStore(); if (!p) return;
    p.then(function(s){ if (s){ try { s.set(lang); window.dispatchEvent(new Event('languagechange')); } catch(e) {} } });
  }
  var n = 0, iv = setInterval(function(){ apply(); if (++n > 30) clearInterval(iv); }, 120);
  apply();
  document.addEventListener('click', function(e){
    try { if (e.target && e.target.closest && e.target.closest('[role=tablist],.tab-nav,[role=tab]')){
      setTimeout(apply, 60); setTimeout(apply, 250);
    }} catch(_) {}
  }, true);
})();
</script>
"""


CSS = """
.gradio-container {max-width: 1080px !important; margin: auto !important;}
.brand-header { position: relative; }
.brand-box { background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #7e22ce 100%);
  padding: 24px 28px; border-radius: 16px; margin: 8px 0 16px 0;
  box-shadow: 0 10px 30px rgba(109,40,217,0.35); color: white; text-align: center; }
.brand-title { font-size: 1.9em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { font-size: 1em; opacity: 0.9; margin-bottom: 14px; }
.brand-credits { font-size: 0.9em; opacity: 0.95; }
.brand-credits a { color:#fbbf24 !important; text-decoration:none !important; font-weight:600 !important; background:none !important; padding:0 !important; }
.brand-credits a:hover { text-decoration: underline !important; }
.device-badge { display:inline-block; background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:999px; font-size:0.85em; margin-top:10px; }
.lang-switcher { position:absolute; top:12px; right:16px; display:flex; gap:6px; align-items:flex-start; z-index:50; }
.lang-btn { background:rgba(255,255,255,0.18); color:white !important; padding:5px 10px; border-radius:8px;
  font-size:0.82em; text-decoration:none !important; font-weight:600; display:inline-flex; flex-direction:column;
  align-items:center; justify-content:center; gap:2px; line-height:1; white-space:nowrap; min-width:44px; cursor:pointer; }
.lang-btn:hover { background:rgba(255,255,255,0.3); }
.lang-btn img { margin:0 !important; vertical-align:middle !important; }
.tabs > div[role="tablist"] > button, .tab-nav > button { flex:1 !important; text-align:center !important; }
.spk-block { background: rgba(124,58,237,0.06); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:10px; margin:6px 0; }
.donate-wrap { position:relative; display:inline-block; }
.donate-wrap > summary.donate-btn { list-style:none; cursor:pointer; }
.donate-wrap > summary.donate-btn::-webkit-details-marker { display:none; }
.donate-wrap:not([open]) .donate-popover { display:none !important; pointer-events:none; }
.donate-popover { position:absolute; top:calc(100% + 6px); right:0; background:rgba(20,20,28,0.98);
  backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:10px 14px;
  min-width:320px; box-shadow:0 8px 24px rgba(0,0,0,0.4); z-index:999; font-size:13px; line-height:1.5; text-align:left; }
.donate-popover a { color:#c4a3ff !important; text-decoration:none !important; font-weight:600 !important; background:none !important; padding:0 !important; }
.donate-row { display:flex; justify-content:space-between; align-items:center; gap:10px; padding:4px 0; }
.donate-row > span { color:#9ca3af; font-weight:600; font-size:12px; flex:0 0 80px; text-align:left; white-space:nowrap; }
.donate-row > code { flex:1; text-align:left; background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px; font-size:11px; color:#e5e7eb; user-select:all; }
.donate-intro { color:#cbd5e1; font-size:12px; line-height:1.5; margin:0 0 4px 0; }
.donate-sep { height:1px; background:rgba(255,255,255,0.1); margin:6px 0; }
/* Тёмные рамки/инпуты (Gradio Soft на тёмном фоне даёт светлые border-vars) + ползунок */
.gradio-container { --block-border-color: rgba(255,255,255,0.10) !important;
  --border-color-primary: rgba(255,255,255,0.10) !important;
  --input-border-color: rgba(255,255,255,0.10) !important;
  --neutral-200: rgba(255,255,255,0.10) !important; }
.gradio-container .block { border-color: rgba(255,255,255,0.10) !important; }
.gradio-container input[type=range] { accent-color: #7c3aed; }
/* Теги-чипы: ширина по тексту, перенос, без растяжения и обрезки */
.tagbtn { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.tagbtn button { white-space: pre-line !important; line-height: 1.15 !important; height: auto !important; min-height: 0 !important; text-align: center !important; padding: 5px 11px !important; font-size: 0.8em !important; }
.tagbtn button > * { display: block; }
.scrollable-files { max-height: 280px; overflow-y: auto !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px; }
"""


import json as _json
# Официальные описания тегов (bosonai/higgs-audio-v3-tts-4b) — для тултипов и легенды.
TAG_DESC = {
    "affection": "Теплота, нежность", "amusement": "Веселье, игривый смешок", "anger": "Гнев",
    "arousal": "Обострённое желание", "awe": "Благоговение, восхищение", "bitterness": "Горечь",
    "confusion": "Растерянность", "contemplation": "Задумчивость, рефлексия", "contentment": "Спокойное удовлетворение",
    "determination": "Решимость, твёрдость", "disgust": "Отвращение", "elation": "Ликование, радость",
    "enthusiasm": "Энтузиазм, воодушевление", "fear": "Страх", "helplessness": "Беспомощность",
    "longing": "Тоска, томление", "pride": "Гордость, уверенность", "relief": "Облегчение",
    "sadness": "Грусть", "shame": "Стыд", "surprise": "Удивление",
    "speed_very_slow": "Очень медленно (≈0.65×)", "speed_slow": "Медленно (≈0.85×)",
    "speed_fast": "Быстро (≈1.2×)", "speed_very_fast": "Очень быстро (≈1.4×)",
    "pitch_low": "Ниже тон (≈−3 полутона)", "pitch_high": "Выше тон (≈+2.5 полутона)",
    "expressive_high": "Выразительнее", "expressive_low": "Ровнее, монотоннее",
    "pause": "Пауза ≈400–700 мс (по месту)", "long_pause": "Длинная пауза ≈700–1500 мс (по месту)",
    "singing": "Пение", "shouting": "Крик, посыл голоса", "whispering": "Шёпот",
    "cough": "Кашель (звук: кхм)", "laughter": "Смех (ха-ха)", "crying": "Плач (ыыы)",
    "screaming": "Крик (ааа)", "burping": "Отрыжка", "humming": "Мычание (ммм)",
    "sigh": "Вздох (эх)", "sniff": "Шмыганье носом", "sneeze": "Чихание (апчхи)",
}
TAG_RU = {k: v.split(",")[0].split(" (")[0].strip() for k, v in TAG_DESC.items()}
_CAT_NAMES = {"emotion": "😊 Эмоции (в начало предложения)", "prosody": "🎵 Просодия",
              "style": "🎭 Стиль (в начало)", "sfx": "🔊 Звуки (по месту, рядом со звукоподражанием)"}
TAGS_LEGEND_MD = "\n\n".join(
    f"**{_CAT_NAMES[c]}**\n" + "\n".join(f"- `<|{c}:{v}|>` — {TAG_DESC.get(v, '')}" for v in sorted(dr.WHITELIST[c]))
    for c in ("emotion", "prosody", "style", "sfx")
)
DARK_JS = ("""
() => {
  try {
    document.documentElement.classList.add('dark');
  } catch(e) {}
  try { const u=new URL(window.location);
    if(u.searchParams.get('__theme')!=='dark'){
      u.searchParams.set('__theme','dark'); window.location.replace(u.href); return;
    } } catch(e){}
  const TD = __TAGDESC__;
  const apply = () => document.querySelectorAll('.tagbtn button').forEach(b => { const k=(b.textContent||'').trim().split(/[\\s\\n]+/).find(x => TD[x]); if(k) b.title = TD[k]; });
  apply(); setInterval(apply, 1200);
}
""").replace("__TAGDESC__", _json.dumps(TAG_DESC, ensure_ascii=False))


TTS_EXAMPLES = [
    ["Привет! Это Higgs Audio Studio — локальная озвучка на ста языках."],
    ["<|emotion:elation|>Невероятно, у нас получилось! <|sfx:laughter|>ха-ха-ха!"],
    ["<|style:whispering|>Подойди ближе, я расскажу секрет."],
    ["Hello! This model speaks over a hundred languages, fully offline."],
]
EXPR_EXAMPLES = [
    ["<|emotion:sadness|>Мне очень жаль, что так вышло. <|prosody:long_pause|> Но мы справимся."],
    ["<|emotion:anger|>Сколько можно это терпеть?! <|sfx:sigh|>эх…"],
    ["<|style:shouting|>Поднажми! Финиш уже совсем близко!"],
]
POD_TOPICS = [["Плюсы и минусы локального ИИ дома"], ["Как нейросети меняют музыку"],
              ["Будущее голосовых ассистентов"]]
BOOK_EXAMPLES = [["Старый маяк молчал уже много лет. — Здесь кто-нибудь есть? — крикнул Том, "
                  "поднимаясь по скрипучей лестнице. Ответом была лишь тишина."]]




# ----------------------------------------------------------------------------
# Хелперы
# ----------------------------------------------------------------------------
_OUT_FORMAT = "mp3"  # как в VoxCPM2: компактный mp3 по умолчанию
# формат → (контейнер soundfile, subtype). MP3/OGG/FLAC/WAV поддержаны libsndfile 0.14 в сборке.
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
    """Пресет → подставить его аудио + транскрипт (как VoxCPM2/Qwen3-TTS)."""
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
    """'Speaker N: текст' (или Диктор/Голос/[N]) → [(speaker_id, text)]; нераспознанное → Speaker 0."""
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
    """Длинный текст → куски (абзацы; длинные — по предложениям) для long-form."""
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




def _speak(text, ref_audio=None, ref_text=None, **kw):
    """Короткий текст — одним проходом, длинный — long-form с переносом голоса."""
    chunks = _chunk(text)
    if len(chunks) <= 1:
        return eng.generate(text, ref_audio=ref_audio, ref_text=ref_text, **kw)
    return eng.synth_longform(chunks, ref_audio=ref_audio, ref_text=ref_text, **kw)




def cb_load_cloud():
    """Список облачных голосов (HF dataset Slait/russia_voices)."""
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
    """Скачать ВСЮ облачную коллекцию (Slait/russia_voices, 700+ голосов)."""
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


def save_tts_settings(text, temp, top_p, top_k, max_new, seed):
    update_gui_config("t_text", text)
    update_gui_config("t_temp", temp)
    update_gui_config("t_top_p", top_p)
    update_gui_config("t_top_k", top_k)
    update_gui_config("t_max", max_new)
    update_gui_config("t_seed", seed)


def save_expr_settings(text, auto):
    update_gui_config("e_text", text)
    update_gui_config("e_auto", auto)


def save_clone_settings(text, preset, auto, temp, top_p, seed):
    update_gui_config("c_text", text)
    update_gui_config("c_preset", preset)
    update_gui_config("c_auto", auto)
    update_gui_config("c_temp", temp)
    update_gui_config("c_top_p", top_p)
    update_gui_config("c_seed", seed)


def save_long_clone_settings(text, preset, max_chars, gap, merge, temp, top_p, seed, auto):
    update_gui_config("lc_text", text)
    update_gui_config("lc_preset", preset)
    update_gui_config("lc_max_chars", max_chars)
    update_gui_config("lc_gap", gap)
    update_gui_config("lc_merge", merge)
    update_gui_config("lc_temp", temp)
    update_gui_config("lc_top_p", top_p)
    update_gui_config("lc_seed", seed)
    update_gui_config("lc_auto", auto)


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




def cb_tts(text, model, auto, temperature, top_p, top_k, max_new, seed):
    eng.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    sr, wav = _speak(text, temperature=temperature, top_p=top_p, top_k=top_k,
                     max_new_tokens=max_new, seed=seed)
    p = _save(sr, wav, "tts")
    return p, text




def cb_enrich(text, model):
    return dr.enrich(text, label=model)




def cb_expr(text, model, auto):
    eng.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    sr, wav = _speak(text)
    p = _save(sr, wav, "expr")
    return p, text




def cb_clone(text, model, auto, ref_audio, ref_text, preset, temperature, top_p, seed):
    eng.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    ref = ref_audio or (voice_path(preset) if preset and preset != OWN_FILE else None)
    sr, wav = _speak(text, ref_audio=ref, ref_text=ref_text, temperature=temperature, top_p=top_p, seed=seed)
    p = _save(sr, wav, "clone")
    return p, text




def cb_test_llm_connection(api_url, api_key, api_model):
    import urllib.request
    import urllib.error
    import json
    

    url = (api_url or "http://127.0.0.1:1234/v1").rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
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




def cb_long_clone(text, model, auto, ref_audio, ref_text, preset, temperature, top_p, seed, max_chars, gap, merge, api_url, api_key, api_model, system_prompt, progress=gr.Progress()):
    eng.clear_cancel()
    import soundfile as sf
    import numpy as np
    from datetime import datetime


    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"long_clone_{stamp}"
    out_dir.mkdir(exist_ok=True)


    ref = ref_audio or (voice_path(preset) if preset and preset != OWN_FILE else None)


    chunks = _chunk(text, max_chars=int(max_chars))
    if not chunks:
        yield "Текст пуст / Text is empty", None, [], ""
        return


    device, name, vram = eng.detect_device()
    log_lines = [
        f"Устройство озвучки / Voice device: {name}",
        f"Разбито на {len(chunks)} частей. Папка: output/long_clone_{stamp}"
    ]
    yield "\n".join(log_lines), None, [], ""


    generated_files = []
    wav_chunks = []
    processed_chunks = []
    sr = 24000


    for i, chunk in enumerate(chunks):
        if eng.cancelled():
            log_lines.append("\n⏹ Остановлено / Stopped.")
            yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks)
            return


        progress((i) / len(chunks), desc=f"{i + 1}/{len(chunks)} · Часть")
        log_lines.append(f"✓ Синтез части {i + 1}/{len(chunks)}: '{chunk[:40]}...'")
        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks)


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
            chunk_sr, chunk_wav = eng.generate(
                chunk_to_speak,
                ref_audio=ref,
                ref_text=ref_text,
                temperature=temperature,
                top_p=top_p,
                seed=seed
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


        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks)


    merged_path = None
    if merge and wav_chunks and not eng.cancelled():
        log_lines.append("\nСклеивание фрагментов...")
        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks)
        

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
        merged_path = str(merged_filename)
        

        _save_path = _save(sr, merged_wav, f"long_clone_merged")
        if _save_path:
            generated_files.insert(0, _save_path)
            

        log_lines.append(f"Склеено и сохранено: {merged_filename.name}")
        yield "\n".join(log_lines), _save_path, generated_files, "\n\n".join(processed_chunks)
    else:
        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks)


    progress(1.0, desc="Готово")
    log_lines.append("\n🎉 Готово / Done!")
    

    if merged_path and wav_chunks:
        yield "\n".join(log_lines), _save_path, generated_files, "\n\n".join(processed_chunks)
    elif wav_chunks:
        yield "\n".join(log_lines), (generated_files[0] if generated_files else None), generated_files, "\n\n".join(processed_chunks)
    else:
        yield "\n".join(log_lines), None, generated_files, "\n\n".join(processed_chunks)




def cb_podcast_script(topic, num, model):
    return dr.write_podcast(topic, int(num), label=model)




def cb_book_markup(text, num, model):
    return dr.cast_audiobook(text, int(num), label=model)




def cb_multi_synth(script, a0, a1, a2, a3, t0, t1, t2, t3, progress=gr.Progress()):
    """Парсим 'Speaker N:' → синтез каждой реплики голосом диктора N → склейка с нормализацией.
    eng._concat выравнивает громкость спикеров (LUFS −16) + пик-лимит, иначе один тише другого."""
    eng.clear_cancel()
    audios = [a0, a1, a2, a3]
    texts = [t0, t1, t2, t3]
    turns = [(sid, txt) for sid, txt in parse_script(script) if txt.strip()]
    if not turns:
        return None
    chunks = []
    for i, (sid, txt) in enumerate(turns):
        if eng.cancelled():
            break
        progress((i + 1) / len(turns), desc=f"{i + 1}/{len(turns)} · Speaker {sid}")
        ref = audios[sid] if 0 <= sid < MAX_SPK else None
        rt = (texts[sid] if 0 <= sid < MAX_SPK else None) or None
        _, wav = _speak(txt, ref_audio=ref, ref_text=rt)
        if wav is not None and len(wav):
            chunks.append(wav)
    final = eng._concat(chunks, gap=0.3)
    p = _save(eng.SR, final, "multi")
    return p




def cb_batch(texts, model, auto, progress=gr.Progress()):
    eng.clear_cancel()
    lines = [t.strip() for t in (texts or "").splitlines() if t.strip()]
    device, name, vram = eng.detect_device()
    log = [f"Устройство озвучки / Voice device: {name}", f"Начало пакетной обработки ({len(lines)} строк)"]
    paths = []
    for i, line in enumerate(lines):
        if eng.cancelled():
            yield "\n".join(log) + "\n\n⏹ Остановлено / Stopped.", paths
            return
        progress((i + 1) / max(len(lines), 1), desc=f"{i + 1}/{len(lines)}")
        if auto:
            line = dr.enrich(line, model)
        sr, wav = _speak(line)
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
    """4 блока диктора (пресет + аудио + транскрипт), показ по слайдеру. Возвращает (slider, audios, texts)."""
    with gr.Row():
        num = gr.Slider(2, MAX_SPK, value=2, step=1, label=T("num_speakers"))
        refresh = gr.Button(T("refresh_voices"), size="sm", scale=0)
    choices = [OWN_FILE] + scan_voices()
    blocks, audios, texts, pres = [], [], [], []
    # Блоки дикторов — вертикально, друг под другом (как в Qwen3-TTS Multi-speaker).
    # Side-by-side в gr.Row давал мигание: тяжёлые waveform конкурировали по ширине.
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




def build():
    llm_cfg = load_llm_config()
    gui_cfg = load_gui_config()
    with gr.Blocks(title=APP_NAME) as demo:
        gr.HTML(T("brand_header_html"))
        model_dd = gr.Dropdown(MODEL_CHOICES, value=gui_cfg.get("model_dd", dr.DEFAULT_MODEL), label=T("director_model"))
        model_dd.change(lambda x: update_gui_config("model_dd", x), [model_dd], None)
        cpu_cb = gr.Checkbox(label=T("cpu_only_label"), value=gui_cfg.get("cpu_only", False))
        def on_cpu_change(val):
            update_gui_config("cpu_only", val)
            eng.set_cpu_mode(val)
            dr.set_cpu_mode(val)
            eng.unload_tts(force=True)
            dr.unload_llm(force=True)
            return f"Режим изменен: {'CPU' if val else 'GPU'}"
        cpu_cb.change(on_cpu_change, [cpu_cb], None)


        keep_cb = gr.Checkbox(label=T("keep_vram_label"), value=gui_cfg.get("keep_vram", False))
        def on_keep_change(val):
            update_gui_config("keep_vram", val)
            if not val:
                eng.unload_tts(force=True)
                dr.unload_llm(force=True)
            return f"Память: {'сохранять' if val else 'выгружать'}"
        keep_cb.change(on_keep_change, [keep_cb], None)
        quant_dd = gr.Dropdown([("bf16 — макс. качество (дефолт)", "bf16"),
                                ("4-bit (nf4) ⚗️ эксперим. — качество ниже", "4bit"),
                                ("8-bit ⚗️ эксперим. — качество ниже", "8bit")],
                               value="bf16", label=T("quant"), info=T("quant_info"))
        quant_dd.change(lambda p: eng.set_precision(p), [quant_dd], None)
        fmt_dd = gr.Radio(["mp3", "wav", "flac", "ogg"], value=gui_cfg.get("out_format", "mp3"), label=T("out_format"))
        def on_format_change(f):
            set_out_format(f)
            update_gui_config("out_format", f)
        fmt_dd.change(on_format_change, [fmt_dd], None)


        with gr.Tabs(selected=gui_cfg.get("active_tab_index", 0)) as tabs:
            # 1. Озвучка
            with gr.Tab(T("tab_tts"), id=0):
                with gr.Row():
                    with gr.Column():
                        t_text = gr.Textbox(label=T("text"), placeholder=T("ph_text"), lines=4, value=gui_cfg.get("t_text", ""))
                        t_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
                        with gr.Accordion(T("advanced"), open=False):
                            t_temp = gr.Slider(0.0, 1.5, gui_cfg.get("t_temp", 1.0), step=0.05, label="Temperature")
                            t_top_p = gr.Slider(0.1, 1.0, gui_cfg.get("t_top_p", 0.95), step=0.01, label="Top-p")
                            t_top_k = gr.Slider(0, 1026, gui_cfg.get("t_top_k", 50), step=1, label="Top-k (0=off)")
                            t_max = gr.Slider(64, 4096, gui_cfg.get("t_max", 2048), step=64, label=T("max_tokens"))
                            t_seed = gr.Number(gui_cfg.get("t_seed", -1), label=T("seed"), precision=0)
                        t_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("t_auto", False))
                        t_btn = gr.Button(T("generate"), variant="primary", size="lg")
                        t_stop = gr.Button(T("stop"), variant="stop")
                    t_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
                gr.Examples(TTS_EXAMPLES, inputs=[t_text], label=T("examples"))
                t_upload.upload(load_text_file, [t_upload], [t_text])
                t_btn.click(save_tts_settings, [t_text, t_temp, t_top_p, t_top_k, t_max, t_seed], None)
                ev_tts = t_btn.click(cb_tts, [t_text, model_dd, t_auto, t_temp, t_top_p, t_top_k, t_max, t_seed],
                                     [t_out, t_text])
                t_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_tts])


            # 2. Экспрессия + Режиссёр
            with gr.Tab(T("tab_expr"), id=1):
                e_text = gr.Textbox(label=T("text"), placeholder=T("ph_text"), lines=5, value=gui_cfg.get("e_text", ""))
                e_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
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
                e_enrich.click(save_expr_settings, [e_text, e_auto], None)
                e_enrich.click(cb_enrich, [e_text, model_dd], [e_text])
                e_btn.click(save_expr_settings, [e_text, e_auto], None)
                ev_expr = e_btn.click(cb_expr, [e_text, model_dd, e_auto], [e_out, e_text])
                e_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_expr])


            # 3. Клонирование
            with gr.Tab(T("tab_clone"), id=2):
                with gr.Row():
                    with gr.Column():
                        c_text = gr.Textbox(label=T("text"), placeholder=T("ph_clone"), lines=3, value=gui_cfg.get("c_text", ""))
                        c_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
                        c_preset = gr.Dropdown([OWN_FILE] + scan_voices(), value=gui_cfg.get("c_preset", OWN_FILE), label=T("voice_preset"))
                        c_refresh = gr.Button(T("refresh"), size="sm")
                        c_ref = gr.Audio(label=T("ref_voice"), type="filepath", sources=["upload", "microphone"])
                        c_ref_text = gr.Textbox(label=T("ref_text"), lines=2, placeholder=T("ph_clone_tr"))
                        c_tr_btn = gr.Button(T("transcribe_btn"), size="sm")
                        c_temp = gr.Slider(0.0, 1.5, gui_cfg.get("c_temp", 1.0), step=0.05, label="Temperature")
                        c_top_p = gr.Slider(0.1, 1.0, gui_cfg.get("c_top_p", 0.95), step=0.01, label="Top-p")
                        c_seed = gr.Number(gui_cfg.get("c_seed", -1), label=T("seed"), precision=0)
                        c_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("c_auto", False))
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
                c_btn.click(save_clone_settings, [c_text, c_preset, c_auto, c_temp, c_top_p, c_seed], None)
                ev_clone = c_btn.click(cb_clone, [c_text, model_dd, c_auto, c_ref, c_ref_text, c_preset, c_temp, c_top_p, c_seed],
                                       [c_out, c_text])
                c_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_clone])


            # 3.5 Длинное клонирование
            with gr.Tab(T("tab_long_clone"), id=3):
                with gr.Row():
                    with gr.Column():
                        lc_text = gr.Textbox(label=T("text"), placeholder=T("ph_clone"), lines=5, value=gui_cfg.get("lc_text", ""))
                        lc_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
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
                        lc_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("lc_auto", False))
                        

                        with gr.Accordion(T("llm_settings_title"), open=False):
                            lc_api_url = gr.Textbox(label=T("api_url_label"), value=llm_cfg["api_url"])
                            lc_api_key = gr.Textbox(label=T("api_key_label"), value=llm_cfg["api_key"], type="password")
                            lc_api_model = gr.Textbox(label=T("api_model_label"), value=llm_cfg["api_model"])
                            lc_system_prompt = gr.Textbox(label=T("system_prompt_label"), value=llm_cfg["system_prompt"], lines=4)
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
                lc_btn.click(save_long_clone_settings, [lc_text, lc_preset, lc_max_chars, lc_gap, lc_merge, lc_temp, lc_top_p, lc_seed, lc_auto], None)
                ev_long_clone = lc_btn.click(cb_long_clone, [lc_text, model_dd, lc_auto, lc_ref, lc_ref_text, lc_preset, lc_temp, lc_top_p, lc_seed, lc_max_chars, lc_gap, lc_merge, lc_api_url, lc_api_key, lc_api_model, lc_system_prompt],
                                             [lc_log, lc_out, lc_files, lc_processed])
                lc_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_long_clone])


            # 4. Подкаст (мульти-спикер, формат Speaker N:)
            with gr.Tab(T("tab_pod")):
                gr.Markdown(T("pod_hint"))
                gr.Markdown(T("pod_format"))
                p_num, p_audios, p_texts = _speaker_blocks()
                p_topic = gr.Textbox(label=T("topic"), placeholder=T("ph_topic"), lines=2)
                gr.Examples(POD_TOPICS, inputs=[p_topic], label=T("examples"))
                p_model = gr.Dropdown(MODEL_CHOICES, value=dr.DEFAULT_MODEL, label=T("director_model"))
                p_script_btn = gr.Button(T("make_script"), variant="secondary")
                p_script = gr.Textbox(label=T("script"), placeholder=T("ph_script"), lines=9)
                p_btn = gr.Button(T("synth"), variant="primary", size="lg")
                p_stop = gr.Button(T("stop"), variant="stop")
                p_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
                p_script_btn.click(cb_podcast_script, [p_topic, p_num, p_model], [p_script])
                ev_pod = p_btn.click(cb_multi_synth, [p_script] + p_audios + p_texts, p_out)
                p_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_pod])


            # 5. Аудиокнига (мульти-спикер: Speaker 0 — рассказчик)
            with gr.Tab(T("tab_book"), id=5):
                gr.Markdown(T("book_hint"))
                gr.Markdown(T("book_format"))
                b_num, b_audios, b_texts = _speaker_blocks()
                b_text = gr.Textbox(label=T("book_text"), placeholder=T("ph_book"), lines=6, value=gui_cfg.get("b_text", ""))
                b_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
                gr.Examples(BOOK_EXAMPLES, inputs=[b_text], label=T("examples"))
                b_model = gr.Dropdown(MODEL_CHOICES, value=gui_cfg.get("model_dd", dr.DEFAULT_MODEL), label=T("director_model"))
                b_markup = gr.Button(T("markup"), variant="secondary")
                b_script = gr.Textbox(label=T("script"), placeholder=T("ph_script"), lines=9)
                b_btn = gr.Button(T("synth"), variant="primary", size="lg")
                b_stop = gr.Button(T("stop"), variant="stop")
                b_out = gr.Audio(label=T("result"), type="filepath", autoplay=True)
                b_upload.upload(load_text_file, [b_upload], [b_text])
                b_markup.click(save_book_settings, [b_text], None)
                b_markup.click(cb_book_markup, [b_text, b_num, b_model], [b_script])
                b_btn.click(save_book_settings, [b_text], None)
                ev_book = b_btn.click(cb_multi_synth, [b_script] + b_audios + b_texts, b_out)
                b_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_book])


            # 6. Пакет
            with gr.Tab(T("tab_batch"), id=6):
                bt_text = gr.Textbox(label=T("batch_text"), placeholder=T("ph_batch"), lines=6, value=gui_cfg.get("bt_text", ""))
                bt_upload = gr.UploadButton(T("upload_txt"), file_types=[".txt"], size="sm")
                bt_auto = gr.Checkbox(label=T("auto_enrich"), value=gui_cfg.get("bt_auto", False))
                bt_btn = gr.Button(T("generate"), variant="primary", size="lg")
                bt_stop = gr.Button(T("stop"), variant="stop")
                bt_log = gr.Textbox(label=T("log"), lines=8)
                bt_files = gr.Files(label=T("result"), elem_classes=["scrollable-files"])
                bt_upload.upload(load_text_file, [bt_upload], [bt_text])
                bt_btn.click(save_batch_settings, [bt_text, bt_auto], None)
                ev_batch = bt_btn.click(cb_batch, [bt_text, model_dd, bt_auto], [bt_log, bt_files])
                bt_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_batch])


        def on_tab_select(evt: gr.SelectData):
            update_gui_config("active_tab_index", evt.index)
        tabs.select(on_tab_select, None, None)

    return demo




def prewarm():
    """Компиляция обоих путей (обычный + клон) в ГЛАВНОМ потоке до старта сервера —
    первая компиляция в рабочем потоке gradio роняет процесс (особенно на клон-пути)."""
    if eng._MOCK:
        return
    try:
        print("[prewarm] прогрев + компиляция в главном потоке (~1-3 мин, разово)...", flush=True)
        eng.generate("Прогрев системы озвучки.")
        vs = scan_voices()
        if vs:
            eng.generate("Прогрев клонирования голоса.", ref_audio=voice_path(vs[0]))
        print("[prewarm] готово — генерации будут быстрыми и стабильными", flush=True)
    except Exception as e:
        print(f"[prewarm] пропущен ({e})", flush=True)




if __name__ == "__main__":
    print(f"[{APP_NAME}] {DEVICE_INFO}")
    # Sync initial CPU mode from config
    cfg = load_gui_config()
    cpu_val = cfg.get("cpu_only", False)
    eng.set_cpu_mode(cpu_val)
    dr.set_cpu_mode(cpu_val)
    # Sync initial output format
    set_out_format(cfg.get("out_format", "mp3"))
    prewarm()
    demo = build().queue(default_concurrency_limit=1)
    

    # Запускаем сервер Gradio в фоновом режиме, чтобы получить точный локальный URL
    _, local_url, _ = demo.launch(
        server_port=None,
        inbrowser=False,
        prevent_thread_lock=True,
        i18n=I18N, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
        css=CSS, js=DARK_JS, head=HEAD_SCRIPT, show_error=True
    )
    

    # Открываем браузер с принудительно включенной темной темой и русским языком по умолчанию
    if not eng._MOCK and os.environ.get("NO_AUTO_BROWSER", "").lower() not in ("1", "true", "yes"):
        import webbrowser
        url = f"{local_url}?__lang=ru&__theme=dark"
        print(f"[UI] Opening browser with theme: {url}", flush=True)
        webbrowser.open(url)
        

    # Блокируем главный поток для работы сервера
    demo.block_thread()
