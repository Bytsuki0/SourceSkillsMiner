$UsersFile   = "github_users.txt"
$MainConfig  = "config_main.ini"
$ConfigFile  = "config.ini"
$Script      = "ScoringSys.py"
$VenvActivate = "D:/Code/SourceSkillsMiner/win_venv/Scripts/Activate.ps1"

# --- Checks ---
if (-not (Test-Path $UsersFile)) {
    Write-Error "Error: $UsersFile not found."; exit 1
}
if (-not (Test-Path $MainConfig)) {
    Write-Error "Error: $MainConfig not found."; exit 1
}
if (-not (Test-Path $Script)) {
    Write-Error "Error: $Script not found."; exit 1
}
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Error: venv activation script not found at $VenvActivate"; exit 1
}

# --- Activate virtual environment ---
& $VenvActivate
Write-Host "Virtual environment activated."

# --- Read token from config_main.ini ---
$TokenLine = Select-String -Path $MainConfig -Pattern "^\s*token\s*=" | Select-Object -First 1
if (-not $TokenLine) {
    Write-Error "Error: Could not read token from $MainConfig."; exit 1
}
$Token = ($TokenLine.Line -split "=", 2)[1].Trim()
Write-Host "Token loaded from $MainConfig."

# --- Loop through users ---
Get-Content $UsersFile | ForEach-Object {
    $username = $_.Trim()

    # Skip empty lines and comments
    if ([string]::IsNullOrWhiteSpace($username) -or $username.StartsWith("#")) {
        return
    }

    Write-Host "--------------------------------------------"
    Write-Host "Processing user: $username"

    # Create config.ini without BOM using .NET directly
    $content = "[github]`nusername = $username`ntoken = $Token`n"
    [System.IO.File]::WriteAllText($ConfigFile, $content, [System.Text.UTF8Encoding]::new($false))

    Write-Host "config.ini created for: $username"

    # Run ScoringSys.py and wait for it to finish
    python $Script
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        Write-Warning "$Script exited with code $ExitCode for user: $username"
    } else {
        Write-Host "Finished running $Script for user: $username"
    }
}

Write-Host "--------------------------------------------"
Write-Host "All users processed."
