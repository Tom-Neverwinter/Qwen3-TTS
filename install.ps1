<#
    .SYNOPSIS
    Qwen3-TTS Environment Installer with Hardware Detection

    .DESCRIPTION
    Automatically detects your hardware (GPU, CUDA) and installs:
    - Python 3.12
    - PyTorch with appropriate CUDA version
    - Flash Attention (if compatible GPU detected)
    - Qwen-TTS and all dependencies

    .EXAMPLE
    PS> .\install.ps1
#>

param(
    [Switch]$Force = $false,
    [Switch]$SkipPython = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Qwen3-TTS Environment Installer" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Hardware Detection
# ============================================================================

function Test-NvidiaGPU {
    Write-Host "[1/7] Detecting NVIDIA GPU..." -ForegroundColor Yellow
    try {
        $nvidiaCheck = nvidia-smi --query-gpu=name, compute_cap --format=csv, noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaCheck) {
            $gpuInfo = $nvidiaCheck -split ","
            $gpuName = $gpuInfo[0].Trim()
            $computeCap = $gpuInfo[1].Trim()
            
            Write-Host "  ✓ Found: $gpuName" -ForegroundColor Green
            Write-Host "  ✓ Compute Capability: $computeCap" -ForegroundColor Green
            
            return @{
                HasGPU                 = $true
                Name                   = $gpuName
                ComputeCapability      = $computeCap
                SupportsFlashAttention = ([double]$computeCap -ge 8.0)
            }
        }
    }
    catch {
        Write-Host "  ℹ No NVIDIA GPU detected" -ForegroundColor Gray
    }
    
    return @{
        HasGPU                 = $false
        SupportsFlashAttention = $false
    }
}

function Get-CUDAVersion {
    Write-Host "[2/7] Detecting CUDA version..." -ForegroundColor Yellow
    try {
        $cudaVersion = nvidia-smi | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
        if ($cudaVersion) {
            Write-Host "  ✓ CUDA Version: $cudaVersion" -ForegroundColor Green
            
            # Determine PyTorch CUDA version
            $majorVersion = [int]($cudaVersion -split '\.')[0]
            if ($majorVersion -ge 12) {
                return "cu121"  # Use CUDA 12.1 for PyTorch
            }
            elseif ($majorVersion -eq 11) {
                return "cu118"
            }
            else {
                return "cpu"
            }
        }
    }
    catch {
        Write-Host "  ℹ CUDA not detected, will use CPU version" -ForegroundColor Gray
    }
    
    return "cpu"
}

# ============================================================================
# Python 3.12 Installation
# ============================================================================

function Install-Python312 {
    Write-Host "[3/7] Checking Python 3.12..." -ForegroundColor Yellow
    
    $python312Paths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Python312\python.exe"
    )
    
    foreach ($path in $python312Paths) {
        if (Test-Path $path) {
            Write-Host "  ✓ Python 3.12 found at: $path" -ForegroundColor Green
            return $path
        }
    }
    
    if ($SkipPython) {
        Write-Host "  ✗ Python 3.12 not found and -SkipPython specified" -ForegroundColor Red
        throw "Python 3.12 is required. Please install it manually or run without -SkipPython"
    }
    
    Write-Host "  ℹ Python 3.12 not found. Downloading installer..." -ForegroundColor Yellow
    
    $installerUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    $installerPath = "$env:TEMP\python-3.12.10-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "  ✓ Downloaded Python 3.12.10 installer" -ForegroundColor Green
        
        Write-Host "  ℹ Installing Python 3.12.10 (this may take a minute)..." -ForegroundColor Yellow
        Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_test=0" -Wait
        
        Remove-Item $installerPath -Force
        
        $installedPath = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
        if (Test-Path $installedPath) {
            Write-Host "  ✓ Python 3.12 installed successfully" -ForegroundColor Green
            return $installedPath
        }
        else {
            throw "Python installation completed but executable not found"
        }
    }
    catch {
        Write-Host "  ✗ Failed to install Python 3.12: $_" -ForegroundColor Red
        throw
    }
}

# ============================================================================
# Virtual Environment Setup
# ============================================================================

function New-VirtualEnvironment {
    param([string]$PythonPath)
    
    Write-Host "[4/7] Setting up virtual environment..." -ForegroundColor Yellow
    
    $venvPath = ".\venv"
    
    if ((Test-Path $venvPath) -and -not $Force) {
        Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
        return $venvPath
    }
    
    if (Test-Path $venvPath) {
        Write-Host "  ℹ Removing existing venv (Force mode)..." -ForegroundColor Yellow
        Remove-Item -Path $venvPath -Recurse -Force
    }
    
    Write-Host "  ℹ Creating virtual environment..." -ForegroundColor Yellow
    & $PythonPath -m venv $venvPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
        return $venvPath
    }
    else {
        throw "Failed to create virtual environment"
    }
}

# ============================================================================
# Package Installation
# ============================================================================

function Install-PyTorch {
    param(
        [string]$VenvPath,
        [string]$CudaVersion
    )
    
    Write-Host "[5/7] Installing PyTorch..." -ForegroundColor Yellow
    
    $pipPath = "$VenvPath\Scripts\pip.exe"
    
    if ($CudaVersion -eq "cpu") {
        Write-Host "  ℹ Installing CPU-only PyTorch..." -ForegroundColor Yellow
        & $pipPath install torch torchvision torchaudio
    }
    else {
        Write-Host "  ℹ Installing PyTorch with CUDA $CudaVersion support..." -ForegroundColor Yellow
        & $pipPath install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CudaVersion"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PyTorch installed successfully" -ForegroundColor Green
    }
    else {
        throw "Failed to install PyTorch"
    }
}

function Install-FlashAttention {
    param(
        [string]$VenvPath,
        [bool]$SupportsFlashAttention
    )
    
    Write-Host "[6/7] Installing Flash Attention..." -ForegroundColor Yellow
    
    if (-not $SupportsFlashAttention) {
        Write-Host "  ℹ Skipping Flash Attention (GPU compute capability < 8.0)" -ForegroundColor Gray
        return
    }
    
    $pipPath = "$VenvPath\Scripts\pip.exe"
    
    # Use precompiled wheel for Windows
    $flashAttnUrl = "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl"
    
    Write-Host "  ℹ Installing precompiled Flash Attention wheel..." -ForegroundColor Yellow
    & $pipPath install $flashAttnUrl
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Flash Attention installed successfully" -ForegroundColor Green
    }
    else {
        Write-Host "  ⚠ Flash Attention installation failed (non-critical)" -ForegroundColor Yellow
    }
}

function Install-QwenTTS {
    param([string]$VenvPath)
    
    Write-Host "[7/7] Installing Qwen-TTS and dependencies..." -ForegroundColor Yellow
    
    $pipPath = "$VenvPath\Scripts\pip.exe"
    
    Write-Host "  ℹ Installing packages..." -ForegroundColor Yellow
    & $pipPath install qwen-tts gradio soundfile requests
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ All dependencies installed successfully" -ForegroundColor Green
    }
    else {
        throw "Failed to install Qwen-TTS dependencies"
    }
}

# ============================================================================
# Main Installation Flow
# ============================================================================

try {
    # Detect hardware
    $gpuInfo = Test-NvidiaGPU
    $cudaVersion = if ($gpuInfo.HasGPU) { Get-CUDAVersion } else { "cpu" }
    
    # Install Python 3.12
    $pythonPath = Install-Python312
    
    # Create virtual environment
    $venvPath = New-VirtualEnvironment -PythonPath $pythonPath
    
    # Install packages
    Install-PyTorch -VenvPath $venvPath -CudaVersion $cudaVersion
    Install-FlashAttention -VenvPath $venvPath -SupportsFlashAttention $gpuInfo.SupportsFlashAttention
    Install-QwenTTS -VenvPath $venvPath
    
    # Summary
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "  Installation Complete! ✓" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Configuration Summary:" -ForegroundColor Cyan
    Write-Host "  • Python: 3.12" -ForegroundColor White
    Write-Host "  • PyTorch: CUDA $cudaVersion" -ForegroundColor White
    Write-Host "  • Flash Attention: $(if ($gpuInfo.SupportsFlashAttention) { 'Enabled' } else { 'Disabled (GPU < 8.0)' })" -ForegroundColor White
    Write-Host "  • GPU: $(if ($gpuInfo.HasGPU) { $gpuInfo.Name } else { 'None (CPU mode)' })" -ForegroundColor White
    Write-Host ""
    Write-Host "To run Qwen3-TTS, execute:" -ForegroundColor Cyan
    Write-Host "  .\RUN_QWEN3.bat" -ForegroundColor Yellow
    Write-Host ""
    
}
catch {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Red
    Write-Host "  Installation Failed ✗" -ForegroundColor Red
    Write-Host "=====================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    exit 1
}
