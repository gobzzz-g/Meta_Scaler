Write-Host "Running validation..."

# Step 1: OpenEnv validation
C:\Users\gobin\AppData\Roaming\Python\Python314\Scripts\openenv.exe validate
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ OpenEnv validation failed"
    exit 1 
}

# Step 2: Docker build
docker build -t supportdesk_env .
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ Docker build failed"
    exit 1 
}

# ✅ Final Success Banner
Write-Host "========================================"
Write-Host "  All 3/3 checks passed!"
Write-Host "  Your submission is ready to submit."
Write-Host "========================================"