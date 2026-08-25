"""Движок FireRedTTS3 (Qwen3 1.7B + DiT Flow Matching + RedAE + CAM++).
Обеспечивает быстрое (10 шагов DiT), стабильное (без галлюцинаций и залипаний)
клонирование голоса на 24 языках (включая русский), Voice Design (создание голосов по описанию)
и Speech Editing (редактирование слов и темпа/тона/громкости речи).
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("FireRedTTS3-Engine")

SR = 24000
_MOCK = bool(os.environ.get("HIGGS_UI_MOCK"))
_CPU_MODE = False

_forced_precision = None  # "bf16" / "int8" / "fp32"
_forced_variant = "fireredtts3_base"  # "fireredtts3_base" / "fireredtts3_instruct"

_ACTIVE_BUNDLE = None
_CANCEL = False

LANGUAGE_CHOICES = [
    "auto",
    "Russian",
    "English",
    "Chinese",
    "Cantonese",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "Arabic",
    "Turkish",
    "Indonesian",
    "Portuguese",
    "Italian",
    "Dutch",
    "Vietnamese",
    "German",
    "Ukrainian",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
    "ZH_Anhui", "ZH_Fujian", "ZH_Gansu", "ZH_Guizhou", "ZH_Hebei",
    "ZH_Henan", "ZH_Hubei", "ZH_Hunan", "ZH_Jiangxi", "ZH_Liaoning",
    "ZH_Minnan", "ZH_Ningxia", "ZH_Shaanxi", "ZH_Shandong", "ZH_Shanghai",
    "ZH_Shanxi", "ZH_Sichuan", "ZH_Tianjin", "ZH_Wenzhou", "ZH_Wu", "ZH_Yunnan",
]


def set_cpu_mode(val: bool) -> None:
    global _CPU_MODE
    _CPU_MODE = bool(val)


def detect_device() -> tuple[str, str, float]:
    if _CPU_MODE:
        return "cpu", "CPU", 0.0
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return "cuda", p.name, p.total_memory / 1e9
    except Exception:
        pass
    return "cpu", "CPU", 0.0


def device_info() -> str:
    if _MOCK:
        return "MOCK UI (без модели)"
    try:
        dev, name, vram = detect_device()
    except Exception:
        return "CPU"
    return f"{name} | VRAM {vram:.1f} ГБ" if dev == "cuda" else "CPU (медленно)"


def set_precision(p: Optional[str]) -> None:
    """Выбор точности: 'bf16' / 'int8' / 'fp32'."""
    global _forced_precision
    if p in ("bf16", "int8", "fp32"):
        _forced_precision = p
    else:
        _forced_precision = None
    unload_tts()


def set_variant(v: Optional[str]) -> None:
    """Выбор варианта: 'fireredtts3_base' / 'fireredtts3_instruct'."""
    global _forced_variant
    if v in ("fireredtts3_base", "fireredtts3_instruct"):
        _forced_variant = v
    else:
        _forced_variant = "fireredtts3_base"
    unload_tts()


def request_cancel() -> None:
    global _CANCEL
    _CANCEL = True
    print("[firered] STOP ОТМЕНА — прерываю генерацию", flush=True)


def clear_cancel() -> None:
    global _CANCEL
    _CANCEL = False


def cancelled() -> bool:
    return _CANCEL


def is_keep_vram() -> bool:
    try:
        import json
        cfg_path = Path(__file__).parent.absolute() / "gui_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f).get("keep_vram", False)
    except Exception:
        pass
    return False


def unload_tts(force: bool = False) -> None:
    """Выгрузить FireRedTTS3 из VRAM."""
    global _ACTIVE_BUNDLE
    if is_keep_vram() and not force:
        return
    if _ACTIVE_BUNDLE is not None:
        try:
            from firered_core.loader import unload_firered_bundle
            unload_firered_bundle(_ACTIVE_BUNDLE, reason="engine unload", hard=True)
        except Exception as e:
            logger.warning("Error unloading bundle: %s", e)
        _ACTIVE_BUNDLE = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_tts(variant: Optional[str] = None, precision: Optional[str] = None) -> Any:
    global _ACTIVE_BUNDLE
    if _MOCK:
        return "MOCK"

    target_variant = variant or _forced_variant or "fireredtts3_base"
    target_prec = precision or _forced_precision or "bf16"

    repo_label = "FireRedTTS3-bf16"
    if target_prec == "int8":
        repo_label = "FireRedTTS3-int8"
    elif target_prec == "fp32":
        repo_label = "FireRedTTS3-fp32"

    if _ACTIVE_BUNDLE is not None:
        if getattr(_ACTIVE_BUNDLE, "variant", "") == ("base" if target_variant == "fireredtts3_base" else "instruct"):
            return _ACTIVE_BUNDLE
        else:
            unload_tts(force=True)

    import torch
    from firered_core.loader import ensure_fasttext_model, load_firered_bundle
    from firered_core.native import set_cancel_hook

    set_cancel_hook(cancelled)

    # Скачивание FastText для авто-определения языка
    ensure_fasttext_model(download_if_missing=True)

    dev_str, dev_name, _ = detect_device()
    print(f"[firered] Загрузка {target_variant} ({repo_label}) на {dev_name}...", flush=True)

    bundle = load_firered_bundle(
        repo_choice=repo_label,
        variant=target_variant,
        dtype_name="auto" if dev_str == "cuda" else "fp32",
        device_name=dev_str,
        attention="auto",
        download_if_missing=True,
    )
    _ACTIVE_BUNDLE = bundle
    return bundle


def _audio_to_tensor(audio_path: str, max_duration: float = 6.0) -> tuple[Any, int]:
    import soundfile as sf
    import torch
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if max_duration > 0 and len(data) > int(sr * max_duration):
        data = data[:int(sr * max_duration)]
    waveform = torch.from_numpy(data.T)  # [channels, samples]
    return waveform, sr


def trim_trailing_artifacts(wav: np.ndarray, sr: int = SR, threshold_ratio: float = 0.04, min_speech_sec: float = 0.25, pad_sec: float = 0.15) -> np.ndarray:
    """Удаляет галлюцинации, тишину и артефакты диффузии в хвосте аудио."""
    if wav is None or len(wav) == 0:
        return wav
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    num_frames = (len(wav) - frame_len) // hop_len
    if num_frames <= 0:
        return wav

    energies = np.array([np.sqrt(np.mean(wav[i * hop_len : i * hop_len + frame_len] ** 2)) for i in range(num_frames)])
    max_e = float(np.max(energies))
    if max_e < 1e-4:
        return wav

    v_thresh = max(0.004, max_e * threshold_ratio)
    voiced_indices = np.where(energies > v_thresh)[0]
    if len(voiced_indices) == 0:
        return wav

    speech_end_frame = voiced_indices[-1]
    for i in range(len(voiced_indices) - 1):
        gap = (voiced_indices[i + 1] - voiced_indices[i]) * 0.010
        if gap > 0.35:  # пауза тишины более 350мс после произнесения фразы
            speech_dur = (voiced_indices[i] - voiced_indices[0]) * 0.010
            if speech_dur >= min_speech_sec or voiced_indices[0] * 0.010 < 0.2:
                rem_max = float(np.max(energies[voiced_indices[i + 1]:]))
                if rem_max < max_e * 0.45 or gap > 0.55:
                    speech_end_frame = voiced_indices[i]
                    break

    end_sample = min(len(wav), (speech_end_frame * hop_len) + frame_len + int(sr * pad_sec))
    trimmed = wav[:end_sample].copy()

    fade_len = int(sr * 0.015)
    if len(trimmed) > fade_len:
        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        trimmed[-fade_len:] *= fade

    return trimmed


def _max_gen_steps(max_audio_seconds: float, sentence_len: int = 0) -> int:
    if sentence_len > 0:
        est_sec = min(float(max_audio_seconds), max(1.2, float(sentence_len) * 0.08 + 1.0))
        return max(3, int(round(est_sec * 25.0 / 4.0)))
    return max(3, int(round(float(max_audio_seconds) * 25.0 / 4.0)))


def _find_default_voice(detected_lang: str) -> Optional[str]:
    voices_dir = Path(__file__).parent / "voices"
    if not voices_dir.exists():
        return None

    lang_lower = (detected_lang or "").lower()
    candidates = []
    if "rus" in lang_lower:
        candidates = ["RU_AK8.wav", "RU_Female_YandexAlisa.mp3", "RU_Male_AbdulovV.mp3"]
    elif "eng" in lang_lower:
        candidates = ["English_Female.wav", "English_Female2.wav", "EN_Female_Anime_Girl.mp3"]

    for cand in candidates:
        cand_path = voices_dir / cand
        if cand_path.exists():
            return str(cand_path)

    for p in sorted(voices_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in (".wav", ".mp3"):
            return str(p)

    return None


def generate(
    text: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    language: str = "auto",
    n_timesteps: int = 10,
    inference_cfg: float = 2.0,
    stop_threshold: float = 0.5,
    seed: int = -1,
    max_audio_seconds: float = 64.0,
    do_tn: bool = True,
    do_split: bool = True,
    cross_fade_ms: float = 50.0,
    variant: Optional[str] = None,
    **kwargs: Any,
) -> tuple[int, np.ndarray]:
    """Озвучить текст (Zero-shot Voice Clone или дефолтный голос). Возвращает (SR, np.float32[L])."""
    text = (text or "").strip()
    if not text:
        return SR, np.zeros(0, np.float32)

    # Удаляем чужие теги режиссера HiggsAudio <|cat:val|> если они есть
    import re
    text = re.sub(r"<\|[^>]+:.*?\|>", "", text).strip()
    if not text:
        return SR, np.zeros(0, np.float32)

    if _MOCK:
        n = int(SR * 1.4)
        return SR, (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32)

    import torch
    from firered_core import native

    # Для многоязычного клонирования (русский и др.) предпочтительна база
    selected_variant = variant or ("fireredtts3_base" if ref_audio else _forced_variant)
    bundle = get_tts(variant=selected_variant)

    if seed is not None and int(seed) >= 0:
        native.fix_seed(int(seed))

    clean_text_full, detected_lang, sentences = bundle.frontend.apply(
        text,
        language=None if language == "auto" else language,
        do_tn=bool(do_tn),
        do_split=bool(do_split),
        tokenize=lambda s: native.measure_tokens(bundle, s),
    )

    t0 = time.time()
    print(f"[firered] >> синтез ({detected_lang}, {len(sentences)} предл.): {clean_text_full[:80]}...", flush=True)

    prompt_latents = None
    prompt_audio_len = 0
    spk_emb = None
    prompt_text = ""

    if ref_audio and os.path.exists(ref_audio):
        # Пользователь передал аудио для клонирования
        waveform, in_sr = _audio_to_tensor(ref_audio, max_duration=6.0)
        prompt_text = (ref_text or "").strip()
        if bundle.variant == "base":
            spk_emb = native.speaker_embedding(bundle, waveform, in_sr)

        # Проверяем, совпадает ли язык референса с целевым языком.
        # В FireRedTTS3 in-context prompt_latents требуют совпадения языка референса и текста.
        # Для кросс-языкового клонирования (напр. русский голос -> английский текст)
        # используется чистый CAM++ speaker embedding без смешивания языковых токенов.
        is_same_lang = True
        if prompt_text:
            try:
                prompt_lang = bundle.frontend.detect(prompt_text)
                if prompt_lang != detected_lang:
                    is_same_lang = False
                    print(f"[firered] Кросс-языковое клонирование ({prompt_lang} -> {detected_lang}): используется CAM++ speaker embedding.", flush=True)
            except Exception:
                pass

        if is_same_lang and prompt_text:
            prompt_latents, prompt_audio_len = native.tokenize_prompt_audio(bundle, waveform, in_sr)
        else:
            prompt_latents = torch.zeros(1, 0, bundle.core.redae_dim, device=bundle.device)
            prompt_audio_len = 0
            prompt_text = ""
    else:
        # Режим обычной озвучки без пользовательского аудио:
        # Используем спикер-эмбеддинг CAM++ дефолтного голоса, но БЕЗ prompt_latents (чистый синтез с нуля)
        def_audio = _find_default_voice(detected_lang)
        if def_audio and os.path.exists(def_audio) and bundle.variant == "base":
            waveform, in_sr = _audio_to_tensor(def_audio, max_duration=5.0)
            spk_emb = native.speaker_embedding(bundle, waveform, in_sr)
        elif bundle.variant == "base":
            spk_emb = torch.zeros(1, 512, device=bundle.device)

        prompt_latents = torch.zeros(1, 0, bundle.core.redae_dim, device=bundle.device)
        prompt_audio_len = 0
        prompt_text = ""

    segments: list[torch.Tensor] = []
    gen_audio_sr = SR

    for idx, sentence in enumerate(sentences):
        if _CANCEL:
            print("[firered] Генерация отменена пользователем.", flush=True)
            break

        max_steps = _max_gen_steps(max_audio_seconds, len(sentence))

        if bundle.variant == "base":
            segment, gen_audio_sr = native.base_clone_one(
                bundle,
                text=sentence,
                language=detected_lang,
                prompt_text=prompt_text or "",
                prompt_latents=prompt_latents,
                prompt_audio_len=prompt_audio_len,
                spk_emb=spk_emb,
                stop_threshold=stop_threshold,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed if seed >= 0 else 0,
                max_gen_steps=max_steps,
            )
        else:
            segment, gen_audio_sr = native.instruct_clone_one(
                bundle,
                text=sentence,
                prompt_text=prompt_text or "",
                prompt_latents=prompt_latents,
                stop_threshold=stop_threshold,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed if seed >= 0 else 0,
                max_gen_steps=max_steps,
            )
        segments.append(segment.cpu())

    if not segments or _CANCEL:
        return SR, np.zeros(0, np.float32)

    gen_audio = segments[0]
    if len(segments) > 1:
        fade_len = int(cross_fade_ms / 1000.0 * gen_audio_sr)
        for seg in segments[1:]:
            gen_audio = native.cross_fade(gen_audio, seg, fade_len)

    wav_np = trim_trailing_artifacts(gen_audio.squeeze().float().numpy(), gen_audio_sr)
    el = time.time() - t0
    sec = len(wav_np) / gen_audio_sr
    print(f"[firered] OK {sec:.1f}с аудио за {el:.1f}с ({sec/max(el, 1e-3):.1f}x реалтайм)", flush=True)
    return gen_audio_sr, wav_np.astype(np.float32)


def voice_design(
    instruction: str,
    text: str,
    language: str = "auto",
    text_temperature: float = 0.7,
    text_top_p: float = 0.8,
    text_top_k: int = 20,
    text_repetition_penalty: float = 1.0,
    n_timesteps: int = 10,
    inference_cfg: float = 1.2,
    seed: int = -1,
    max_audio_seconds: float = 64.0,
    do_tn: bool = True,
    do_split: bool = True,
    cross_fade_ms: float = 50.0,
) -> tuple[int, np.ndarray, str]:
    """Синтез голоса по словесному описанию (Voice Design через FireRedTTS3-Instruct)."""
    text = (text or "").strip()
    instruction = (instruction or "").strip()
    if not text or not instruction:
        return SR, np.zeros(0, np.float32), ""

    if _MOCK:
        n = int(SR * 1.4)
        return SR, (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32), "Mock Voice Plan"

    import torch
    from firered_core import native

    bundle = get_tts(variant="fireredtts3_instruct")

    if seed is not None and int(seed) >= 0:
        native.fix_seed(int(seed))

    clean_text_full, detected_lang, sentences = bundle.frontend.apply(
        text,
        language=None if language == "auto" else language,
        do_tn=bool(do_tn),
        do_split=bool(do_split),
        tokenize=lambda s: native.measure_tokens(bundle, s),
    )

    t0 = time.time()
    print(f"[firered-design] >> Voice Design: '{instruction}' -> {len(sentences)} предл.", flush=True)

    max_steps = _max_gen_steps(max_audio_seconds, len(clean_text_full))
    segments: list[torch.Tensor] = []
    gen_audio_sr = SR
    voice_plan = ""

    for idx, sentence in enumerate(sentences):
        if _CANCEL:
            break
        segment, gen_audio_sr, segment_plan = native.voice_design_one(
            bundle,
            instruction=instruction,
            text=sentence,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            seed=seed if seed >= 0 else 0,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            text_repetition_penalty=text_repetition_penalty,
            max_gen_steps=max_steps,
        )
        segments.append(segment.cpu())
        if idx == 0:
            voice_plan = segment_plan

    if not segments or _CANCEL:
        return SR, np.zeros(0, np.float32), voice_plan

    gen_audio = segments[0]
    if len(segments) > 1:
        fade_len = int(cross_fade_ms / 1000.0 * gen_audio_sr)
        for seg in segments[1:]:
            gen_audio = native.cross_fade(gen_audio, seg, fade_len)

    wav_np = trim_trailing_artifacts(gen_audio.squeeze().float().numpy(), gen_audio_sr)
    el = time.time() - t0
    sec = len(wav_np) / gen_audio_sr
    print(f"[firered-design] OK {sec:.1f}с аудио за {el:.1f}с. План голоса: {voice_plan[:100]}", flush=True)
    return gen_audio_sr, wav_np.astype(np.float32), voice_plan


def semantic_edit(
    audio_path: str,
    instruction: str,
    n_timesteps: int = 10,
    inference_cfg: float = 1.2,
    stop_threshold: float = 0.5,
    seed: int = -1,
    max_audio_seconds: float = 64.0,
) -> tuple[int, np.ndarray, str]:
    """Семантическое редактирование речи (вставка/удаление/замена слов)."""
    if not audio_path or not os.path.exists(audio_path):
        return SR, np.zeros(0, np.float32), ""

    import torch
    from firered_core import native

    bundle = get_tts(variant="fireredtts3_instruct")
    if seed is not None and int(seed) >= 0:
        native.fix_seed(int(seed))

    waveform, in_sr = _audio_to_tensor(audio_path)
    latents_in, _ = native.tokenize_prompt_audio(bundle, waveform, in_sr)
    max_steps = _max_gen_steps(max_audio_seconds, len(instruction))

    segment, gen_audio_sr, edited_text = native.semantic_edit_one(
        bundle,
        instruction=instruction,
        latents_in=latents_in,
        n_timesteps=n_timesteps,
        inference_cfg=inference_cfg,
        seed=seed if seed >= 0 else 0,
        stop_threshold=stop_threshold,
        max_gen_steps=max_steps,
    )
    wav_np = trim_trailing_artifacts(segment.squeeze().cpu().float().numpy(), gen_audio_sr)
    return gen_audio_sr, wav_np.astype(np.float32), edited_text


def acoustic_edit(
    audio_path: str,
    mode: str = "speed",
    value: float = 1.2,
    custom_instruction: str = "",
    n_timesteps: int = 10,
    inference_cfg: float = 1.2,
    stop_threshold: float = 0.5,
    seed: int = -1,
    max_audio_seconds: float = 64.0,
) -> tuple[int, np.ndarray]:
    """Акустическое редактирование речи (изменение скорости / высоты тона / громкости)."""
    if not audio_path or not os.path.exists(audio_path):
        return SR, np.zeros(0, np.float32)

    import torch
    from firered_core import native

    bundle = get_tts(variant="fireredtts3_instruct")
    if seed is not None and int(seed) >= 0:
        native.fix_seed(int(seed))

    if custom_instruction.strip():
        instruction = custom_instruction.strip()
    elif mode == "speed":
        instruction = f"adjust the speed to {value:.1f}x"
    elif mode == "volume":
        instruction = f"adjust the volume to {value:.1f}x"
    else:
        steps = int(round(value))
        instruction = f"shift the pitch by {steps} step" + ("s" if abs(steps) != 1 else "")

    waveform, in_sr = _audio_to_tensor(audio_path)
    latents_in, _ = native.tokenize_prompt_audio(bundle, waveform, in_sr)
    max_steps = _max_gen_steps(max_audio_seconds)

    segment, gen_audio_sr = native.acoustic_edit_one(
        bundle,
        instruction=instruction,
        latents_in=latents_in,
        n_timesteps=n_timesteps,
        inference_cfg=inference_cfg,
        seed=seed if seed >= 0 else 0,
        stop_threshold=stop_threshold,
        max_gen_steps=max_steps,
    )
    wav_np = trim_trailing_artifacts(segment.squeeze().cpu().float().numpy(), gen_audio_sr)
    return gen_audio_sr, wav_np.astype(np.float32)


# ----------------------------------------------------------------------------
# Нормализация громкости (EBU R128 LUFS) и микширование
# ----------------------------------------------------------------------------
TARGET_LUFS = -16.0
_PEAK_CEIL = 10 ** (-1.0 / 20)
_MAX_GAIN = 10 ** (20.0 / 20)


def _loudness_normalize(x: np.ndarray, sr: int = SR) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    try:
        import pyloudnorm as pyln
        loud = pyln.Meter(sr).integrated_loudness(x)
        if np.isfinite(loud):
            gain = 10 ** ((TARGET_LUFS - loud) / 20)
            return (x * min(gain, _MAX_GAIN)).astype(np.float32)
    except Exception:
        pass
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-6:
        return x
    return (x * min((10 ** (-20.0 / 20)) / rms, _MAX_GAIN)).astype(np.float32)


def _peak_limit(x: np.ndarray, ceil: float = _PEAK_CEIL) -> np.ndarray:
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    return (x * (ceil / peak)).astype(np.float32) if peak > ceil else x


def _concat(chunks: list[np.ndarray], gap: float = 0.3, normalize: bool = True) -> np.ndarray:
    chunks = [c for c in chunks if c is not None and len(c)]
    if not chunks:
        return np.zeros(0, np.float32)
    if normalize:
        chunks = [_loudness_normalize(c) for c in chunks]
    sil = np.zeros(int(SR * gap), np.float32)
    out = []
    for i, c in enumerate(chunks):
        if i:
            out.append(sil)
        out.append(c)
    mix = np.concatenate(out)
    return _peak_limit(mix) if normalize else mix


def synth_longform(paragraphs: list[str], ref_audio: Optional[str] = None, ref_text: Optional[str] = None, **kw: Any) -> tuple[int, np.ndarray]:
    """Озвучка длинного текста по абзацам с сохранением голоса."""
    chunks = []
    chain_ref, chain_txt = ref_audio, ref_text
    paras = [p for p in paragraphs if p and p.strip()]

    import soundfile as sf

    for i, para in enumerate(paras):
        if _CANCEL:
            print(f"[firered] STOP остановлено на чанке {i + 1}/{len(paras)}", flush=True)
            break
        print(f"[firered] лонг-форм: чанк {i + 1}/{len(paras)}", flush=True)
        _, a = generate(para, ref_audio=chain_ref, ref_text=chain_txt, **kw)
        chunks.append(a)
        if i == 0 and not _MOCK and ref_audio is None and len(a):
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.close()
            sf.write(f.name, a, SR)
            chain_ref, chain_txt = f.name, para

    return SR, _concat(chunks)


def synth_turns(turns: list[dict[str, Any]], gap: float = 0.4, **kw: Any) -> tuple[int, np.ndarray]:
    """Мульти-спикерная озвучка диалогов/подкаста."""
    chunks = []
    for t in turns:
        if _CANCEL:
            break
        txt = t.get("text", "").strip()
        if txt:
            _, a = generate(txt, ref_audio=t.get("ref_audio"), ref_text=t.get("ref_text"), **kw)
            chunks.append(a)
    return SR, _concat(chunks, gap)
