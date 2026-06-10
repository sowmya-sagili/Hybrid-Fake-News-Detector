# setup_env.ps1 - Complete environment setup for Hybrid Fake News Detector
# Run from project root: .\setup_env.ps1

Set-StrictMode -Off
$ErrorActionPreference = "Continue"

Write-Host "=== Fake News Detector - Environment Setup ===" -ForegroundColor Cyan
Write-Host ""

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = "$ProjectDir\.venv\bin\python.exe"

# Ensure MSYS2 tools are on PATH
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH

# --- Step 1: SSL Certificate ---
Write-Host "[1/6] Setting up SSL certificates..." -ForegroundColor Yellow
$certPath = "C:\msys64\ucrt64\etc\ssl\cert.pem"
if (-not (Test-Path $certPath)) {
    $certs = Get-ChildItem -Path Cert:\LocalMachine\Root
    $pemContent = ""
    foreach ($cert in $certs) {
        $base64 = [Convert]::ToBase64String($cert.RawData, [Base64FormattingOptions]::InsertLineBreaks)
        $pemContent += "-----BEGIN CERTIFICATE-----`n$base64`n-----END CERTIFICATE-----`n"
    }
    Set-Content -Path $certPath -Value $pemContent -Encoding ASCII
    Write-Host "  Created cert.pem with $($certs.Count) Windows CA certs" -ForegroundColor Green
} else {
    Write-Host "  cert.pem already exists" -ForegroundColor Green
}

# --- Step 2: Enable system site-packages in venv ---
Write-Host "[2/6] Enabling system site-packages..." -ForegroundColor Yellow
$pvenvCfg = "$ProjectDir\.venv\pyvenv.cfg"
if (Test-Path $pvenvCfg) {
    (Get-Content $pvenvCfg) -replace "include-system-site-packages = false", "include-system-site-packages = true" | Set-Content $pvenvCfg
    Write-Host "  pyvenv.cfg updated" -ForegroundColor Green
}

# --- Step 3: Install certifi first ---
Write-Host "[3/6] Installing certifi..." -ForegroundColor Yellow
& $VenvPython -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org certifi | Out-Null
$certBundle = & $VenvPython -c "import certifi; print(certifi.where())" 2>&1
Write-Host "  CA bundle: $certBundle" -ForegroundColor Green

# Set headers path for C extensions
$env:C_INCLUDE_PATH = "C:\msys64\ucrt64\include\python3.12"
$env:INCLUDE       = "C:\msys64\ucrt64\include\python3.12"
$env:CFLAGS        = "-I C:\msys64\ucrt64\include\python3.12"

# --- Step 4: Core ML packages ---
Write-Host "[4/6] Installing core ML packages (may take 5-10 minutes to compile)..." -ForegroundColor Yellow
$packages = @("numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.2", "joblib==1.4.2", "threadpoolctl")
foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..." -NoNewline
    $result = & $VenvPython -m pip install --cert $certBundle $pkg 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
    }
}

# --- Step 5: App packages ---
Write-Host "[5/6] Installing app packages..." -ForegroundColor Yellow
$appPkgs = @(
    "streamlit", "pandas", "plotly", "requests", "altair",
    "textblob", "vaderSentiment", "python-dotenv",
    "google-generativeai", "fastapi", "uvicorn", "slowapi",
    "pydantic", "python-multipart"
)
& $VenvPython -m pip install --cert $certBundle @appPkgs 2>&1 | Tee-Object -Variable pipOut
if ($LASTEXITCODE -eq 0) {
    Write-Host "  App packages installed OK" -ForegroundColor Green
} else {
    Write-Host "  Some app packages failed - check output above" -ForegroundColor Yellow
}

# --- Step 6: Train model ---
Write-Host "[6/6] Training ML model..." -ForegroundColor Yellow
if (-not (Test-Path "$ProjectDir\models\model.joblib")) {
    & $VenvPython "$ProjectDir\train_model.py"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Model trained successfully!" -ForegroundColor Green
    } else {
        Write-Host "  Model training failed - check output above" -ForegroundColor Red
    }
} else {
    Write-Host "  Model already exists, skipping training" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the app:" -ForegroundColor White
Write-Host "  .\.venv\bin\Activate.ps1; streamlit run app.py" -ForegroundColor Yellow
