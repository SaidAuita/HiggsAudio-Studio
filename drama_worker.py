"""Воркер Fish Audio Drama 3 (запускается через ComfyUI Python с CUDA 13 / cu130).
Управляется через JSON-RPC по stdin/stdout.
"""
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

# Добавляем пути к ComfyUI и модулям Drama3
COMFY_DIR = r"Z:\ComfyUI\ComfyUI_art_v9\ComfyUI"
CUSTOM_NODES_DIR = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-Drama3")
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)
if CUSTOM_NODES_DIR not in sys.path:
    sys.path.insert(0, CUSTOM_NODES_DIR)

import torch
import torchaudio
import comfy.model_management
import comfy.utils

from drama3.audio import tidy_clip, trim_pauses
from drama3.chunking import CHUNK_BYTES, split_text
from drama3.codec import SAMPLE_RATE, FRAME_LENGTH
from drama3.generate import generate
from drama3.loader import load_codec, load_drama3
from drama3.prompt import build_prompt, with_speaker_tag, speaker_of, as_speaker_0, LINE_SEPARATOR
from qwen3_asr.asr import load_qwen3_asr, transcribe

TAKE_GAP = 0.3
MAX_PAUSE = 0.75

logging.basicConfig(level=logging.INFO, format="[Drama3Worker] %(asctime)s - %(levelname)s - %(message)s")

_loaded_model = None
_loaded_codec = None
_loaded_asr = None

def send_response(data: dict):
    line = json.dumps(data, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

def log(msg: str):
    try:
        sys.stderr.write(f"[Drama3Worker] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass

_ref_cache = {}

def get_cached_reference(codec, asr, ref_audio, ref_text):
    global _ref_cache
    if not ref_audio or not os.path.exists(ref_audio):
        return []
    try:
        mtime = os.path.getmtime(ref_audio)
    except Exception:
        mtime = 0
    key = (os.path.abspath(ref_audio), ref_text or "", mtime)
    if key in _ref_cache:
        return _ref_cache[key]
    
    log(f"Кодирование референса голоса: {ref_audio}")
    codes, raw_wave = encode_audio_clip(codec, ref_audio)
    final_text = ref_text
    if not final_text and asr is not None:
        log("Авто-распознавание текста референса через Qwen3-ASR...")
        try:
            final_text = transcribe(asr, raw_wave, language="auto", context="")
            log(f"Распознан референс: {final_text}")
        except Exception as e:
            log(f"Ошибка ASR: {e}")
            final_text = "speech reference"
    elif not final_text:
        final_text = "speech reference"
    
    ref_list = [(with_speaker_tag(final_text, 0), codes)]
    _ref_cache[key] = ref_list
    return ref_list

def ensure_models(model_name="drama3_int8experts_bf16.safetensors"):
    global _loaded_model, _loaded_codec, _loaded_asr
    
    models_dir = Path(COMFY_DIR) / "models"
    model_path = models_dir / "diffusion_models" / model_name
    codec_path = models_dir / "vae" / "drama3_codec.safetensors"
    asr_path = models_dir / "asr" / "qwen3_asr_0.6b_bf16.safetensors"
    
    if _loaded_codec is None:
        if not codec_path.exists():
            raise FileNotFoundError(f"Кодек не найден: {codec_path}")
        log(f"Загрузка кодека Drama 3 из {codec_path}...")
        _loaded_codec = load_codec(str(codec_path))
        log("Кодек успешно загружен.")

    if _loaded_asr is None:
        if asr_path.exists():
            log(f"Загрузка Qwen3-ASR из {asr_path}...")
            try:
                _loaded_asr = load_qwen3_asr(str(asr_path))
                log("Qwen3-ASR успешно загружен.")
            except Exception as e:
                log(f"Предупреждение: не удалось загрузить ASR: {e}")

    if _loaded_model is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        log(f"Загрузка модели Drama 3 из {model_path} (эксперты в RAM, кеш в VRAM)...")
        # experts_in_ram=True держит 38ГБ в системной RAM и подкачивает кеш в 24ГБ VRAM
        _loaded_model = load_drama3(str(model_path), second_device=None, split=0, experts_in_ram=True, expert_cache_gb=4.0)
        log("Модель Drama 3 успешно загружена!")

    return _loaded_model, _loaded_codec, _loaded_asr

def safe_trim_pauses(waveform: torch.Tensor, sample_rate: int, max_pause: float = 0.75, silence_rms: float = 0.001, head_sec: float = 0.08, tail_sec: float = 0.15) -> torch.Tensor:
    """Умная обрезка пауз с сохранением мягких согласных (H, Kh, Sh) и запасом пре-ролла в 80 мс."""
    SILENCE_WINDOW = 0.05
    window = max(1, int(SILENCE_WINDOW * sample_rate))
    rms = torch.stack([piece.mean() for piece in waveform.pow(2).mean(0).split(window)]).sqrt()
    quiet = (rms < silence_rms).tolist()
    if all(quiet):
        return waveform
    
    first = min(i for i, silent in enumerate(quiet) if not silent)
    lead_in = int(head_sec * sample_rate)
    head_start = max(0, first * window - lead_in)
    
    pieces, held, started = [], 0, False
    for index, silent in enumerate(quiet):
        start = index * window
        if not silent:
            if not started:
                started = True
                pieces.append((head_start, min(start + window, waveform.shape[-1])))
            else:
                pieces.append((start, min(start + window, waveform.shape[-1])))
            held = 0
        elif started:
            keep = min(window, max(0, int(max_pause * sample_rate) - held))
            if keep:
                pieces.append((start, start + keep))
            held += keep

    last = max(i for i, silent in enumerate(quiet) if not silent)
    limit = (last + 1) * window + int(tail_sec * sample_rate)
    kept = [(start, min(end, limit)) for start, end in pieces if start < limit]
    return torch.cat([waveform[:, start:end] for start, end in kept], dim=-1)

@torch.inference_mode()
def decode_codes(codec, codes, gap=0.0, shorten_long_pauses=True):
    takes = [take for take in codes["codes"].split(codes["takes"], -1) if take.shape[-1]]
    if not takes:
        return torch.zeros((1, FRAME_LENGTH))
    
    comfy.model_management.load_models_gpu([codec.patcher])
    device = codec.patcher.load_device
    with comfy.model_management.cuda_device_context(device):
        audio = [codec.model.decode(take[None].to(device), torch.float32).float().cpu() for take in takes]
    silence = torch.zeros((1, round(gap * SAMPLE_RATE)))
    waveform = torch.cat([piece for take in audio for piece in (silence, take)][1:], -1)
    if shorten_long_pauses and codes["codes"].shape[-1]:
        waveform = safe_trim_pauses(waveform, SAMPLE_RATE, MAX_PAUSE)
    return waveform

@torch.inference_mode()
def encode_audio_clip(codec, audio_path: str):
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform[None]
    elif waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        
    waveform = tidy_clip(waveform)
    comfy.model_management.load_models_gpu([codec.patcher])
    device = codec.patcher.load_device
    with comfy.model_management.cuda_device_context(device):
        codes = codec.model.encode(waveform.to(device))[0].cpu()
    return codes, waveform

def save_wav(waveform: torch.Tensor, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim > 1:
        waveform = waveform[0]
    pcm = (waveform.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    with wave.open(out_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)

@torch.inference_mode()
def handle_synth(req: dict):
    text = req.get("text", "").strip()
    if not text:
        return {"status": "error", "error": "Текст пуст"}
        
    ref_audio = req.get("ref_audio")
    ref_text = req.get("ref_text", "").strip()
    seed = int(req.get("seed", 42))
    temperature = float(req.get("temperature", 1.0))
    top_p = float(req.get("top_p", 0.9))
    top_k = int(req.get("top_k", 0))
    max_frames = int(req.get("max_frames", 600))
    out_wav = req.get("out_wav")
    model_name = req.get("model_name", "drama3_int8experts_bf16.safetensors")

    model, codec, asr = ensure_models(model_name)

    references = get_cached_reference(codec, asr, ref_audio, ref_text)

    # Разбивка на чанки если текст длинный
    chunks = split_text(text, CHUNK_BYTES, max_speakers=1) or [text]
    log(f"Генерация {len(chunks)} фрагментов...")

    comfy.model_management.load_models_gpu(model.patchers())
    config = model.config
    separator = model.tokenizer.token_to_id("<|audio_separator|>")
    
    taken, pieces = [], []
    with model.session():
        for index, chunk in enumerate(chunks):
            speaker = speaker_of(chunk)
            said = as_speaker_0(chunk, speaker)
            cast = [(as_speaker_0(t, speaker), c) for t, c in references if speaker_of(t) == speaker]
            own = [turn for turn in taken if speaker_of(turn[0]) == speaker]
            history = own[:1] + own[1:][-1:]
            
            prompt = build_prompt(model.tokenizer, model.config, said, cast, [(as_speaker_0(t, speaker), f) for t, f in history])
            
            def on_frame_cb(step):
                if step % 5 == 0:
                    send_response({
                        "event": "progress",
                        "step": step,
                        "max_frames": max_frames,
                        "chunk": index,
                        "total_chunks": len(chunks),
                        "stage": "generating"
                    })

            chunk_frames = generate(model, prompt, max_frames, temperature, top_k, top_p, seed + index, on_frame=on_frame_cb, lines=LINE_SEPARATOR in chunk)
            audio = chunk_frames[(chunk_frames[:, 0] >= config.semantic_start_token_id) & (chunk_frames[:, 0] <= config.semantic_end_token_id), 1:].T
            pieces.append((index, [], audio))
            if len(chunk_frames):
                taken.append((chunk, chunk_frames))

    pieces.sort(key=lambda p: p[0])
    sound = sum(audio.shape[-1] for _, _, audio in pieces)
    takes = [audio.shape[-1] for _, _, audio in pieces]
    all_codes = torch.cat([audio for _, _, audio in pieces], -1).contiguous()
    
    log(f"Декодирование {sound} аудио-фреймов в WAV...")
    send_response({
        "event": "progress",
        "step": max_frames,
        "max_frames": max_frames,
        "chunk": len(chunks) - 1,
        "total_chunks": len(chunks),
        "stage": "decoding"
    })
    waveform = decode_codes(codec, {"codes": all_codes, "takes": takes}, gap=0.0, shorten_long_pauses=True)
    
    if not out_wav:
        out_wav = os.path.abspath(f"temp/drama3_{int(time.time()*1000)}.wav")
    save_wav(waveform, out_wav)
    
    duration = waveform.shape[-1] / SAMPLE_RATE
    log(f"Синтез завершен! Длительность: {duration:.2f} сек. -> {out_wav}")

    res = {
        "status": "ok",
        "out_wav": out_wav,
        "sample_rate": SAMPLE_RATE,
        "duration": duration,
        "ref_text": ref_text,
    }
    try:
        del waveform, all_codes, pieces, taken
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res

def handle_unload():
    global _loaded_model, _loaded_codec, _loaded_asr
    _loaded_model = None
    _loaded_codec = None
    _loaded_asr = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("Модели выгружены из памяти.")
    return {"status": "ok"}

def main():
    log("Drama 3 Worker запущен и готов к командам.")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            cmd = req.get("cmd")
            
            if cmd == "ping":
                send_response({"status": "ok", "msg": "pong"})
            elif cmd == "synth":
                res = handle_synth(req)
                send_response(res)
            elif cmd == "unload":
                res = handle_unload()
                send_response(res)
            elif cmd == "exit":
                send_response({"status": "ok"})
                break
            else:
                send_response({"status": "error", "error": f"Неизвестная команда: {cmd}"})
        except Exception as e:
            log(f"Исключение в цикле воркера: {e}")
            send_response({"status": "error", "error": str(e)})

if __name__ == "__main__":
    main()

