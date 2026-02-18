$UsersFile    = "github_users.txt"
$MainConfig   = "config_main.ini"
$Script       = "ScoringSys.py"
$VenvPython   = "D:/Code/SourceSkillsMiner/win_venv/Scripts/python.exe"

# Maximum simultaneous instances
$MaxParallel  = 2

# --- Checks ---
if (-not (Test-Path $UsersFile))  { Write-Error "Error: $UsersFile not found.";  exit 1 }
if (-not (Test-Path $MainConfig)) { Write-Error "Error: $MainConfig not found."; exit 1 }
if (-not (Test-Path $Script))     { Write-Error "Error: $Script not found.";     exit 1 }
if (-not (Test-Path $VenvPython)) { Write-Error "Error: Python not found at $VenvPython"; exit 1 }

# --- Read token ---
$TokenLine = Select-String -Path $MainConfig -Pattern "^\s*token\s*=" | Select-Object -First 1
if (-not $TokenLine) { Write-Error "Error: Could not read token from $MainConfig."; exit 1 }
$Token = ($TokenLine.Line -split "=", 2)[1].Trim()
Write-Host "Token loaded from $MainConfig."

# --- Collect users ---
$Users = Get-Content $UsersFile |
    ForEach-Object { $_.Trim() } |
    Where-Object   { -not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#") }

if ($Users.Count -eq 0) { Write-Warning "No users found in $UsersFile."; exit 0 }

Write-Host "Found $($Users.Count) user(s). Running up to $MaxParallel in parallel."
Write-Host "--------------------------------------------"

$AbsScript   = (Resolve-Path $Script).Path
$RunningJobs = @{}

$JobBlock = {
    param(
        [string]$Username,
        [string]$Token,
        [string]$PythonExe,
        [string]$ScriptPath,
        [string]$JobDir
    )

    # Create an isolated working directory for this user
    New-Item -ItemType Directory -Path $JobDir -Force | Out-Null

    # Write config.ini into it — ScoringSys.py reads config.ini from cwd,
    # so no changes to the Python script are needed.
    $content = "[github]`nusername = $Username`ntoken = $Token`n"
    [System.IO.File]::WriteAllText(
        (Join-Path $JobDir "config.ini"),
        $content,
        [System.Text.UTF8Encoding]::new($false)
    )

    Set-Location $JobDir
    & $PythonExe $ScriptPath
    $exit = $LASTEXITCODE

    return [PSCustomObject]@{
        Username = $Username
        ExitCode = $exit
        JobDir   = $JobDir
    }
}

function Wait-ForSlot {
    param([int]$MaxAllowed)
    while ($RunningJobs.Count -ge $MaxAllowed) {
        Start-Sleep -Milliseconds 500
        foreach ($u in @($RunningJobs.Keys)) {
            if ($RunningJobs[$u].State -in 'Completed','Failed','Stopped') {
                Receive-And-Report-Job -Username $u
            }
        }
    }
}

function Receive-And-Report-Job {
    param([string]$Username)
    $job    = $RunningJobs[$Username]
    $result = Receive-Job -Job $job -Wait -ErrorAction SilentlyContinue

    if ($job.State -eq 'Failed') {
        Write-Warning "[$Username] Job FAILED: $($job.ChildJobs[0].JobStateInfo.Reason)"
    } elseif ($result) {
        if ($result.ExitCode -ne 0) {
            Write-Warning "[$Username] Script exited with code $($result.ExitCode)"
        } else {
            Write-Host "[$Username] Finished successfully."
        }
        # Clean up the temporary working directory
        if (Test-Path $result.JobDir) { Remove-Item $result.JobDir -Recurse -Force }
    } else {
        Write-Warning "[$Username] No result returned."
    }

    Remove-Job -Job $job -Force
    $RunningJobs.Remove($Username)
}

# --- Dispatch ---
foreach ($username in $Users) {
    Wait-ForSlot -MaxAllowed $MaxParallel

    # Isolated temp folder per user — avoids any file collision between jobs
    $jobDir = Join-Path ([System.IO.Path]::GetTempPath()) "ssminer_$username"

    Write-Host "Starting job for: $username"
    $job = Start-Job `
        -ScriptBlock  $JobBlock `
        -ArgumentList $username, $Token, $VenvPython, $AbsScript, $jobDir

    $RunningJobs[$username] = $job
}

# --- Drain remaining jobs ---
Write-Host "--------------------------------------------"
Write-Host "All jobs dispatched. Waiting for $($RunningJobs.Count) remaining job(s)..."

while ($RunningJobs.Count -gt 0) {
    Start-Sleep -Milliseconds 500
    foreach ($u in @($RunningJobs.Keys)) {
        if ($RunningJobs[$u].State -in 'Completed','Failed','Stopped') {
            Receive-And-Report-Job -Username $u
        }
    }
}

Write-Host "--------------------------------------------"
Write-Host "All users processed."
