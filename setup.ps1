# FitRoom one-time setup: Python 3.11 venv, CUDA 12.8 PyTorch (RTX 50xx ready), deps, code, models, gallery.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

Step "Installing uv (fast Python installer)"
python -m pip install --user -q uv
$uv = "python -m uv"

Step "Creating Python 3.11 virtual environment"
if (-not (Test-Path .venv\Scripts\python.exe)) { python -m uv venv .venv --python 3.11 }
$py = ".venv\Scripts\python.exe"

Step "Installing PyTorch (CUDA 12.8)"
python -m uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128

Step "Installing app dependencies"
python -m uv pip install --python $py -r requirements.txt

Step "Fetching CatVTON + detectron2 source"
if (-not (Test-Path third_party\CatVTON\model)) {
  git clone --depth 1 https://huggingface.co/spaces/zhengchong/CatVTON third_party/CatVTON
}
if (-not (Test-Path third_party\detectron2_src\detectron2)) {
  git clone --depth 1 https://github.com/facebookresearch/detectron2.git third_party/detectron2_src
}

Step "Downloading model weights (~6 GB)"
$env:HF_HOME = "$PSScriptRoot\models\hf"
& $py scripts\download_models.py

Step "Building sample gallery"
& $py scripts\build_gallery.py

Write-Host "`nSetup complete. Start the app with run.bat" -ForegroundColor Green
