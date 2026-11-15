<div align="center">

# 🎙️ Audio Studio TTS

<!-- super-markdown-toc -->

## Table of Contents

* [Audio Studio TTS](#audio-studio-tts)
  + [Features](#features)
  + [Multi-Engine Architecture](#multi-engine-architecture)
  + [System Requirements](#system-requirements)
    - [Platforms](#platforms)
    - [Memory and Quantization](#memory-and-quantization)
  + [Quick Start](#quick-start)
  + [Batch Clone & Video Translation Workflow](#batch-clone--video-translation-workflow)
  + [Fork Additions and Updates](#fork-additions-and-updates)
  + [Authors & Acknowledgments](#authors--acknowledgments)
  + [License](#license)
  + [Support this Project](#support-this-project)
<!-- /super-markdown-toc -->

**Next-generation local speech synthesis and voice cloning studio: powered by Fish Audio Drama 3 (35B MoE, Hi-Fi 44.1 kHz), Higgs Audio v3.1, and FireRedTTS3. 100% offline, one-click portable runtime.**

[![License](https://img.shields.io/badge/license-Research_%26_Non--Commercial-orange?style=flat-square)](#license)
[![Stars](https://img.shields.io/github/stars/SaidAuita/HiggsAudio-Studio?style=flat-square)](https://github.com/SaidAuita/HiggsAudio-Studio/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/SaidAuita/HiggsAudio-Studio?style=flat-square)](https://github.com/SaidAuita/HiggsAudio-Studio/commits/main)

**[English](README.md)** · **[Русский](README_RU.md)**

<img src="images/Audio%20Studio%20Batch%20clone.png" alt="Audio Studio TTS — Batch Clone Interface" width="850"/>

</div>

---

## Features

* **🎭 Flagship Engine: Fish Audio Drama 3 (35B MoE)** — studio-grade **Hi-Fi 44.1 kHz** speech synthesis, state-of-the-art prosody, breathing, human-like cadence, and direct voice acting. Currently the gold standard and highest-quality local TTS engine for professional voice cloning and video dubbing.
* **🎬 Batch Clone & Video Translation Workflow** — automated sequential synthesis of video subtitle/transcript fragments (`ru-en.txt` ➔ `01.wav`, `02.wav`...), decade navigation (`01-10`, `11-20`...), direct 1-click inspection & re-voicing per fragment, and seamless gap-controlled concatenation (`01-33.wav`).
* **🎙️ Higgs Audio v3.1** — 43 inline emotion tags (`<|emotion:happy|>`), sound effects (`<|sfx:laughter|>`), prosody, and style controls + auto-enrichment via local LLM or External API (LM Studio, Ollama, OpenAI).
* **🔥 FireRedTTS3 (DiT Flow Matching)** — crystal-clear zero-shot cloning, instant **Voice Design** from natural-language prompts («*A warm, confident male voice in his 30s*»), and **Speech Edit** (word replacement, pitch/speed editing).
* **🧬 Zero-Shot Voice Cloning** — clone any voice from a short reference audio clip (3–6 seconds) with automatic transcript recognition (Moonshine ASR). Full support for cross-lingual cloning (e.g. Russian reference ➔ English speech).
* **🧬 Long Voice Cloning** — voice-over for large books and chapters with sentence chunking, step-by-step synthesis, seamless concatenation, customizable gaps, and auto-increment file numbering (`01.mp3`, `02.mp3`...).
* **🎬 Podcast & Audiobook Studio** — multi-speaker casting, script generation, and role attribution with standard EBU R128 loudness leveling (LUFS −16).
* **📦 Batch Processing** — mass synthesis of multiple text lines with live progress tracking.
* **🔇 Intelligent VAD Tail Cleaning** — `trim_trailing_artifacts` filter completely removes vocoder clicks, trailing diffusion noise, and hallucinations on single words and short phrases.
* **🌐 100% Bilingual Interface (RU 🇷🇺 / EN 🇬🇧)** — real-time client-side language switching without page reloads.
* **💾 Full GUI State Persistence** — automatically saves and restores selected engine, sub-tabs, presets, quantization mode (`bf16`, `8-bit`, `4-bit`, `fp32`), and parameters across app restarts.

---

## Multi-Engine Architecture

| Parameter | 🎭 Fish Audio Drama 3 (Primary / Flagship) | 🎙️ Higgs Audio v3.1 | 🔥 FireRedTTS3 |
| :--- | :--- | :--- | :--- |
| **Developer** | Fish Audio | Boson AI | FireRedTeam / Xiaohongshu |
| **Architecture** | 35B MoE Dual-Autoregressive Transformer + Hi-Fi Neural Codec | 4B Autoregressive Transformer + Vocoder | Qwen3 1.7B + 10-step DiT Flow Matching + RedAE + CAM++ |
| **Audio Quality** | **Hi-Fi 44.1 kHz (Studio Grade)** | 24 kHz | 24 kHz |
| **Quantization** | `bf16`, `fp32` (efficient isolated worker + reference caching) | `bf16`, `8-bit (INT8)`, `4-bit (NF4)`, `fp32` | `bf16`, `8-bit (INT8)`, `fp32` |
| **Languages** | Multilingual (Russian, English, Chinese, Japanese, etc.) | 100+ languages (expressive focus) | 24 languages |
| **Key Strengths** | **Unmatched voice cloning fidelity, human-like cadence & breathing, high-bitrate studio sound, zero robotic artifacts** | 43 emotion/SFX tags, theatrical acting, AI Director | Pristine clarity, instant Voice Design, Speech Edit, stability |
| **Best For** | **Video tutorials, course translation, dubbing, high-fidelity audiobooks** | Dramatic acting, emotional storytelling, roleplay | Voice creation from text, speech editing, quick voice cloning |
| **Supported Modes** | TTS, Clone, Long Clone, Batch Clone, Podcast, Book, Batch | TTS, Expressive, Clone, Long Clone, Batch Clone, Podcast, Book, Batch | TTS, Clone, Voice Design, Speech Edit, Long Clone, Batch Clone, Podcast, Book, Batch |

---

## System Requirements

### Platforms

| OS | GPU | Status | Acceleration |
|---|---|---|---|
| Windows 10/11 | NVIDIA RTX 30xx–50xx | ✅ tested | CUDA 12.8 / 12.6 + Triton / SDPA |
| Windows 10/11 | NVIDIA RTX 20xx / GTX 16xx | ✅ supported | CUDA 12.6 / 11.8 |
| Windows / Linux | CPU only | ✅ supported | Streaming CPU synthesis |

### Memory and Quantization

| VRAM | Recommended Setup | Recommended Engines |
|:---|:---|:---|
| **24 GB+** (RTX 3090/4090/5090) | Full precision `bf16` + Director LLM (9–12B in 4-bit) | 🎭 Fish Audio Drama 3, Higgs Audio v3.1, FireRedTTS3 |
| **12–16 GB** (RTX 4070/4080/3060 12G) | `bf16` / `8-bit` + Director LLM (4–9B in 4-bit) | 🎭 Fish Audio Drama 3, Higgs Audio v3.1 (`8-bit`), FireRedTTS3 |
| **8–10 GB** (RTX 3070/3080/4060) | `8-bit` / `4-bit` | FireRedTTS3, Higgs Audio (`8-bit` / `4-bit`), Drama 3 (isolated worker) |
| **CPU** | `💻 Use CPU only` mode | Runs entirely on CPU without GPU |

---

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SaidAuita/HiggsAudio-Studio.git
   cd HiggsAudio-Studio
   ```

2. **Install:**
   * Run `install.bat` and select your GPU CUDA version (CUDA 12.8, 12.6, 11.8 or CPU).
   * The installer configures a portable Python 3.12 environment with PyTorch and all dependencies.

3. **Download Fish Audio Drama 3 (Flagship):**
   * Run PowerShell script:
     ```powershell
     powershell -ExecutionPolicy Bypass -File download_drama3.ps1
     ```
   * Or models will download on first demand into `models/`.

4. **Launch:**
   * Run `run.bat`.
   * The app opens automatically in your browser at `http://127.0.0.1:7860`.

---

## Batch Clone & Video Translation Workflow

Designed specifically for translating YouTube tutorials, educational courses, and commercial video lessons:

* **Automated Sequential Voicing**: Ingests bilingual or monolingual cue scripts (e.g. `ru-en.txt`) and synthesizes individual chunks (`01.wav`, `02.wav`...) saved directly to `output/NUM/`.
* **Interactive Decade Navigation**: Browse large scripts effortlessly by decades (`01-10`, `11-20`, `21-30`...) with 10 direct-access buttons showing live readiness badges (`✅ 21` ready, `⚪ 25` missing) and full replica previews.
* **Instant 1-Click Inspection & Re-Voicing**: Clicking any fragment immediately loads its waveform into the player, populates the text field, sets the range (`From 21` to `To 21`), and enables immediate re-generation if a specific intonation needs fine-tuning.
* **Integrated Audio Merger**: Stitch verified chunks from custom range (`From 01` to `To 33`) into a single output file (e.g. `01-33.wav` with automatic `_01` suffix collision protection) with customizable inter-phrase pause gaps (default 0.5s).
* **Ready-to-Use Script Format & Sample**: A 5-fragment sample script is provided in [`examples/batch_script_sample.txt`](examples/batch_script_sample.txt) supporting cue numbers (`[01.wav]`) and bilingual tags (`RU:` / `EN:`).

---

## Fork Additions and Updates

* **🎭 Fish Audio Drama 3 Flagship Integration (35B MoE, Hi-Fi 44.1kHz)**:
  * Full integration of Fish Audio Drama 3 with studio-grade 44.1kHz sampling rate.
  * Isolated sub-process worker architecture (`drama_worker.py` + `drama_engine.py`) to eliminate CUDA memory fragmentation and OS pipe buffer deadlocks.
  * Smart reference audio caching (`get_cached_reference`) avoiding redundant Qwen3-ASR and codec re-encoding on consecutive batch chunks.
  * One-click model acquisition script `download_drama3.ps1`.
* **🎬 Batch Clone & Educational Video Translation Workflow**:
  * Chunk-by-chunk voicing with interactive decade navigation, live visual status badges, 1-click re-generation, and custom-range audio stitcher.
* **🔥 FireRedTTS3 Integration**:
  * Full support for `FireRedTTS3-bf16`, `FireRedTTS3-fp32`, and `FireRedTTS3-int8`.
  * Integrated **Voice Design** (generating unique voices from text prompts via the Instruct model) and **Speech Edit** (semantic and acoustic audio editing).
  * Automatic cross-lingual cloning using CAM++ speaker embeddings.
* **🔇 End-of-Speech VAD Cleaning (`trim_trailing_artifacts`)**:
  * Eliminates vocoder boundary clicks, trailing diffusion noise, and repeating hallucinations on short sentences and single words.
  * Optimized diffusion step calculation, accelerating short-phrase synthesis up to 3.5×.
* **⚡ Process Isolation for RTX 30xx/40xx/50xx**:
  * Background daemon `director_daemon.py` isolates CUDA contexts between `llama.cpp` and `PyTorch`, preventing crashes.
* **🧬 Long Voice Cloning with Auto-Numbering**:
  * Batch synthesis for audiobook chapters with automatic file numbering (`01`, `02`, `03`...) saved to `output/NUM/`.
* **🌐 Full Bilingual Support (RU / EN)**:
  * 100% translation coverage for all UI elements, tabs, dropdowns, examples, and tooltips.

---

### 💎 QR Code with Custom Shape Logo

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://ph-cu-s.com/ai-code-pro">
        <img src="docs/ai_code_pro.png" width="130" alt="AI Code PRO - Custom Shape Logo QR Code" /><br/>
        <b>AI Code PRO</b>
      </a><br/>
      <sub>Vector Barcodes (18 Standards) & Designer QR Codes for <b>Adobe Illustrator</b></sub><br/><br/>
      <a href="https://ph-cu-s.com/ai-code-pro"><img src="https://img.shields.io/badge/PRO-Illustrator%20Extension-orange.svg" alt="AI Code PRO" /></a>
    </td>
    <td align="center" width="50%">
      <a href="https://ph-cu-s.com/id-code-pro">
        <img src="docs/id_code_pro.png" width="130" alt="ID Code PRO - Custom Shape Logo QR Code" /><br/>
        <b>ID Code PRO</b>
      </a><br/>
      <sub>Vector Barcodes (18 Standards) & Designer QR Codes for <b>Adobe InDesign</b></sub><br/><br/>
      <a href="https://ph-cu-s.com/id-code-pro"><img src="https://img.shields.io/badge/PRO-InDesign%20Extension-purple.svg" alt="ID Code PRO" /></a>
    </td>
  </tr>
</table>

---

## Authors & Acknowledgments

* **timoncool** — author of the original HiggsAudio-Studio project.
* **SaidAuita** — author of this fork, Fish Audio Drama 3 integration, FireRedTTS3 integration, multi-engine architecture, Batch Clone mode, VAD filtering, and localization.
* **Fish Audio** — creators of [Fish Audio Drama 3 / Fish Speech](https://github.com/fishaudio/fish-speech).
* **Boson AI** — creators of [Higgs Audio v3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b).
* **FireRedTeam / Xiaohongshu** — creators of [FireRedTTS3](https://github.com/FireRedTeam/FireRedTTS3).
* **UsefulSensors** — [Moonshine ASR](https://github.com/usefulsensors/moonshine) reference transcription.

---

## License

Wrapper code is open-source. Fish Audio Drama 3, Higgs Audio v3, and FireRedTTS3 model weights are provided under their respective licenses. Voice cloning is allowed strictly with the consent of the voice owner.

---

## Support this Project

If this project is helpful to you and you wish to support its development:
* **USDT (TRC20):** `TBWzmMZWbirvACAtPfoZioAhhwSM4n2ArY`
* **Website:** [Ph-CU-S.com](https://ph-cu-s.com) (Photoshop to ComfyUI Plugin)
