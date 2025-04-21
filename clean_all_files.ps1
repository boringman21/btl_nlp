Get-ChildItem -Path "water_leakage" -Recurse -Filter "*.py" | ForEach-Object {
    Write-Host "Processing: $($_.FullName)"
    $content = Get-Content $_.FullName -Raw
    $content | Set-Content -Path $_.FullName -Encoding UTF8 -NoNewline
} 