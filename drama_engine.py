"""Движок Fish Audio Drama 3 для HiggsAudio-Studio.
Обеспечивает озвучку через флагманскую модель Fish Audio Drama 3 (35B MoE, 44.1 kHz Hi-Fi),
с поддержкой режиссерских тегов эмоций ([whispers], [laughs], [sighs], [angry]),
zero-shot клонирования голоса, авто-распознавания референса через Qwen3-ASR
и высококачественного длинного клонирования.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("Drama3-Engine")

SR = 44100
_MOCK = bool(os.environ.get("HIGGS_UI_MOCK"))
_CANCEL = False

COMFY_PYTHON = r"Z:\ComfyUI\ComfyUI_art_v9\python_embeded\python.exe"
WORKER_SCRIPT = str(Path(__file__).parent / "drama_worker.py")

_worker_proc: Optional[subprocess.Popen] = None

def request_cancel() -> None:
    global _CANCEL
    _CANCEL = True
    print("[drama3] Запрошена отмена синтеза", flush=True)

def cancelled() -> bool:
    return _CANCEL

def reset_cancel() -> None:
    global _CANCEL
    _CANCEL = False

def is_available() -> bool:
    models_dir = Path(r"Z:\ComfyUI\ComfyUI_art_v9\ComfyUI\models")
    model_path = models_dir / "diffusion_models" / "drama3_int8experts_bf16.safetensors"
    codec_path = models_dir / "vae" / "drama3_codec.safetensors"
    return os.path.exists(COMFY_PYTHON) and (model_path.exists() or codec_path.exists())

def _ensure_worker() -> subprocess.Popen:
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        return _worker_proc

    if not os.path.exists(COMFY_PYTHON):
        raise FileNotFoundError(f"Python с поддержкой cu130 не найден: {COMFY_PYTHON}")

    print(f"[drama3] Запуск воркера Drama 3 через {COMFY_PYTHON}...", flush=True)
    os.makedirs("temp", exist_ok=True)
    _worker_log_file = open("temp/drama_worker.log", "a", encoding="utf-8")
    _worker_proc = subprocess.Popen(
        [COMFY_PYTHON, "-u", WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=_worker_log_file,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    # Проверка связи ping
    req = json.dumps({"cmd": "ping"}) + "\n"
    _worker_proc.stdin.write(req)
    _worker_proc.stdin.flush()

    resp_line = _worker_proc.stdout.readline()
    if not resp_line:
        err = ""
        try:
            if os.path.exists("temp/drama_worker.log"):
                with open("temp/drama_worker.log", "r", encoding="utf-8", errors="ignore") as lf:
                    err = "".join(lf.readlines()[-20:])
        except Exception:
            pass
        raise RuntimeError(f"Воркер Drama 3 завершился с ошибкой: {err}")

    resp = json.loads(resp_line.strip())
    if resp.get("status") != "ok":
        raise RuntimeError(f"Ошибка ответа воркера: {resp}")

    print("[drama3] Воркер успешно подключен и готов к работе.", flush=True)
    return _worker_proc

def unload_model() -> None:
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        try:
            req = json.dumps({"cmd": "unload"}) + "\n"
            _worker_proc.stdin.write(req)
            _worker_proc.stdin.flush()
            _worker_proc.stdout.readline()
        except Exception:
            pass
    print("[drama3] Модель Drama 3 выгружена.", flush=True)

def trim_trailing_artifacts(wav: np.ndarray, sr: int = SR, **kwargs) -> np.ndarray:
    """Для Drama 3 очистка пауз уже выполняется нативно в codec/audio (trim_pauses)."""
    return wav

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

def synth(
    text: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    seed: int = 42,
    max_frames: int = 1500,
    model_name: str = "drama3_int8experts_bf16.safetensors",
    **kwargs: Any,
) -> tuple[int, np.ndarray]:
    """Синтезирует речь для реплики с помощью Drama 3."""
    reset_cancel()
    text = (text or "").strip()
    if not text:
        return SR, np.zeros(0, np.float32)

    effective_max_frames = max(max_frames, min(4096, len(text) * 12))

    if _MOCK:
        n = int(SR * 1.5)
        return SR, (0.2 * np.sin(2 * np.pi * 330 * np.arange(n) / SR)).astype(np.float32)

    proc = _ensure_worker()

    tmp_out = os.path.abspath(f"temp/drama3_synth_{int(time.time() * 1000)}.wav")
    os.makedirs(os.path.dirname(tmp_out), exist_ok=True)

    req = {
        "cmd": "synth",
        "text": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text or "",
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed if seed >= 0 else 42,
        "max_frames": effective_max_frames,
        "out_wav": tmp_out,
        "model_name": model_name,
    }

    t0 = time.time()
    print(f"[drama3] Отправка задания в Drama 3: '{text[:60]}...' (seed={seed}, temp={temperature})", flush=True)

    line = json.dumps(req, ensure_ascii=False) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()

    on_progress = kwargs.get("on_progress")
    while True:
        resp_line = proc.stdout.readline()
        if not resp_line:
            global _worker_proc
            _worker_proc = None
            err = ""
            try:
                if os.path.exists("temp/drama_worker.log"):
                    with open("temp/drama_worker.log", "r", encoding="utf-8", errors="ignore") as lf:
                        err = "".join(lf.readlines()[-20:])
            except Exception:
                pass
            raise RuntimeError(f"Воркер Drama 3 прервал выполнение: {err}")

        line_str = resp_line.strip()
        if not line_str:
            continue
        try:
            resp = json.loads(line_str)
        except Exception:
            continue

        if resp.get("event") == "progress":
            if on_progress:
                try:
                    on_progress(
                        resp.get("step", 0),
                        resp.get("max_frames", 1),
                        resp.get("chunk", 0),
                        resp.get("total_chunks", 1),
                        resp.get("stage", "generating")
                    )
                except Exception:
                    pass
            continue
        break

    if resp.get("status") != "ok":
        raise RuntimeError(f"Ошибка генерации Drama 3: {resp.get('error')}")

    out_file = resp.get("out_wav")
    if not os.path.exists(out_file):
        raise FileNotFoundError(f"Файл результата не создан: {out_file}")

    data, in_sr = sf.read(out_file, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    try:
        os.remove(out_file)
    except Exception:
        pass

    el = time.time() - t0
    dur = len(data) / in_sr
    print(f"[drama3] Готово: {dur:.2f} сек аудио за {el:.1f} сек (x{dur/max(0.01, el):.1f} real-time)", flush=True)

    return in_sr, data.astype(np.float32)

def synth_longform(
    paragraphs: list[str],
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    gap: float = 0.3,
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int = 42,
    **kw: Any,
) -> tuple[int, np.ndarray]:
    """Озвучка длинного текста по фрагментам с сохранением консистентности персонажа Drama 3."""
    reset_cancel()
    chunks = []
    paras = [p for p in paragraphs if p and p.strip()]

    chain_ref = ref_audio
    chain_txt = ref_text

    print(f"[drama3] Старт длинного клона Drama 3: {len(paras)} частей", flush=True)

    for i, para in enumerate(paras):
        if _CANCEL:
            print(f"[drama3] Остановлено пользователем на шаге {i + 1}/{len(paras)}", flush=True)
            break

        print(f"[drama3] Часть {i + 1}/{len(paras)}...", flush=True)
        chunk_seed = (seed + i) if seed >= 0 else -1

        sr, a = synth(
            para,
            ref_audio=chain_ref,
            ref_text=chain_txt,
            temperature=temperature,
            top_p=top_p,
            seed=chunk_seed,
            **kw,
        )
        chunks.append(a)

        # Если своего референса не было, первый синтезированный кусочек становится голосовым референсом
        if i == 0 and not _MOCK and not ref_audio and len(a):
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.close()
            sf.write(f.name, a, sr)
            chain_ref, chain_txt = f.name, para

    return SR, _concat(chunks, gap=gap)

