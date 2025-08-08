Param(
  [string]$Profile = "local"  # keep for future (e.g., staging/prod)
)

Write-Host "Pulling images..."
docker compose pull

Write-Host "Recreating containers..."
docker compose up -d

Write-Host "Waiting for API health..."
$ok = $false
foreach ($i in 1..30) {
  try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
    if ($resp.StatusCode -eq 200) { $ok = $true; break }
  } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { Write-Warning "API not healthy yet. Check 'docker compose logs -f api'." } else { Write-Host "API is healthy." }

Write-Host "MLflow UI:    http://127.0.0.1:5500"
Write-Host "API docs:     http://127.0.0.1:8000/docs"
Write-Host "Streamlit UI: http://127.0.0.1:8501"
