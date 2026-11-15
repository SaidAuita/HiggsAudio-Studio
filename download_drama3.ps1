$proxy = "http://127.0.0.1:10809"
$vaeDir = "Z:\ComfyUI\ComfyUI_art_v9\ComfyUI\models\vae"
$asrDir = "Z:\ComfyUI\ComfyUI_art_v9\ComfyUI\models\asr"
$diffDir = "Z:\ComfyUI\ComfyUI_art_v9\ComfyUI\models\diffusion_models"

New-Item -ItemType Directory -Force -Path $vaeDir | Out-Null
New-Item -ItemType Directory -Force -Path $asrDir | Out-Null
New-Item -ItemType Directory -Force -Path $diffDir | Out-Null

$items = @(
    @{
        Label = "[1/3] Drama 3 Codec (~1.6 GB)"
        Url = "https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/resolve/main/drama3_codec.safetensors"
        Out = "$vaeDir\drama3_codec.safetensors"
    },
    @{
        Label = "[2/3] Qwen3-ASR 0.6B (~1.6 GB)"
        Url = "https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI/resolve/main/qwen3_asr_0.6b_bf16.safetensors"
        Out = "$asrDir\qwen3_asr_0.6b_bf16.safetensors"
    },
    @{
        Label = "[3/3] Drama 3 int8 MoE Model (~38.7 GB)"
        Url = "https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/resolve/main/drama3_int8experts_bf16.safetensors"
        Out = "$diffDir\drama3_int8experts_bf16.safetensors"
    }
)

foreach ($item in $items) {
    Write-Host "`n============================================================"
    Write-Host "  $($item.Label)"
    Write-Host "============================================================"
    curl.exe -x $proxy -L -C - $item.Url -o $item.Out --retry 30 --retry-delay 3
}

Write-Host "`n[DONE] Все веса Drama 3 загружены!"

