import sys
import json
import re
import os

# Set localized environment variables for HF cache
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TEMP", os.path.join(script_dir, "temp"))
os.environ.setdefault("TMP", os.path.join(script_dir, "temp"))
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(script_dir, "temp"))
os.environ.setdefault("HF_HOME", os.path.join(script_dir, "models"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(script_dir, "models"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(script_dir, "models"))
os.environ.setdefault("TORCH_HOME", os.path.join(script_dir, "models", "torch"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(script_dir, "cache"))

log_file = open("daemon.log", "w", encoding="utf-8")
def log(msg):
    log_file.write(msg + "\n")
    log_file.flush()

log("Daemon starting...")
from huggingface_hub import hf_hub_download
log("hf_hub_download imported.")
from llama_cpp import Llama
log("llama_cpp imported.")

MODELS = {
    "Qwen3.5-9B · Q4_K_M (дефолт, ~5.5 ГБ)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    "Qwen3.5-4B · Q4_K_M (лёгкая, ~2.5 ГБ)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
}

_llm = None
_cur = None

def load_llm(label, use_cpu=False):
    global _llm, _cur
    # If parameters or mode changed, reload
    if _cur == label and _llm is not None and getattr(load_llm, "_last_cpu", False) == use_cpu:
        return _llm
    _llm = None
    _cur = None
    load_llm._last_cpu = use_cpu
    repo, fname = MODELS[label]
    log(f"Downloading model if needed: {repo}/{fname}")
    path = hf_hub_download(repo, fname)
    log(f"Model path: {path}. Initializing Llama (use_cpu={use_cpu})...")
    gpu_layers = 0 if use_cpu else -1
    _llm = Llama(model_path=path, n_gpu_layers=gpu_layers, n_ctx=8192, verbose=False)
    log("Llama initialized successfully.")
    _cur = label
    return _llm

def _strip_think(txt):
    return re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()

def main():
    log("main() started.")
    while True:
        log("Waiting for line from stdin...")
        line = sys.stdin.readline()
        if not line:
            log("stdin EOF reached. Exiting.")
            break
        log(f"Received line: {line.strip()[:100]}")
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            action = req.get("action")
            log(f"Parsed action: {action}")
            if action == "chat":
                system = req["system"]
                user = req["user"]
                max_new = req.get("max_new", 1024)
                temp = req.get("temp", 0.4)
                label = req.get("label")
                
                log(f"Loading model: {label}")
                use_cpu = req.get("use_cpu", False)
                llm = load_llm(label, use_cpu=use_cpu)
                log("Model loaded. Generating completion...")
                out = llm.create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_new, temperature=temp, top_p=0.9
                )
                log("Completion generated.")
                content = _strip_think(out["choices"][0]["message"].get("content", ""))
                sys.stdout.write(json.dumps({"status": "success", "content": content}) + "\n")
                sys.stdout.flush()
                log("Response sent to stdout.")
            elif action == "unload":
                global _llm, _cur
                _llm = None
                _cur = None
                sys.stdout.write(json.dumps({"status": "success"}) + "\n")
                sys.stdout.flush()
                log("Unloaded model, responded. Exiting cleanly.")
                sys.exit(0)
            else:
                sys.stdout.write(json.dumps({"status": "error", "error": "unknown action"}) + "\n")
                sys.stdout.flush()
                log(f"Unknown action: {action}")
        except Exception as e:
            sys.stdout.write(json.dumps({"status": "error", "error": str(e)}) + "\n")
            sys.stdout.flush()
            log(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
