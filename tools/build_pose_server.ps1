param(
    [string]$Python = "python",
    [string]$DependencyPath = "",
    [string]$ModelDirectory = "",
    [switch]$HandOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputDirectory = Join-Path $projectRoot "dist\jade_pose_server"

if (-not $ModelDirectory) {
    $localModels = Join-Path $projectRoot "pose_server\models"
    $userModels = Join-Path $env:USERPROFILE ".mediapipe\models"
    $ModelDirectory = if (Test-Path (Join-Path $localModels "hand_landmarker.task")) {
        $localModels
    } else {
        $userModels
    }
}
$ModelDirectory = (Resolve-Path -LiteralPath $ModelDirectory).Path
if ($ModelDirectory.TrimEnd('\') -eq $outputDirectory.TrimEnd('\') -or
    $ModelDirectory.StartsWith($outputDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "ModelDirectory must be outside the PyInstaller output directory: $ModelDirectory"
}
$modelFiles = if ($HandOnly) { @("hand_landmarker.task") } else { @("pose_landmarker_full.task", "hand_landmarker.task") }
foreach ($filename in $modelFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $ModelDirectory $filename) -PathType Leaf)) {
        throw "Missing model: $(Join-Path $ModelDirectory $filename)"
    }
}

$previousPythonPath = $env:PYTHONPATH
try {
    if ($DependencyPath) {
        $DependencyPath = (Resolve-Path -LiteralPath $DependencyPath).Path
        $env:PYTHONPATH = if ($previousPythonPath) {
            "$DependencyPath;$previousPythonPath"
        } else {
            $DependencyPath
        }
    }

    & $Python -c "import PyInstaller, mediapipe, cv2, numpy"
    if ($LASTEXITCODE -ne 0) {
        throw "The build Python must have PyInstaller, mediapipe, OpenCV, and NumPy."
    }

    $entryPoint = if ($HandOnly) { "pose_server\hand_only.py" } else { "pose_server\standalone.py" }
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --name jade_pose_server `
        --distpath (Join-Path $projectRoot "dist") `
        --workpath (Join-Path $projectRoot ".build\pyinstaller") `
        --specpath (Join-Path $projectRoot ".build\pyinstaller") `
        --paths $projectRoot `
        --collect-all mediapipe `
        --collect-all cv2 `
        (Join-Path $projectRoot $entryPoint)
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}

$modelsOutput = Join-Path $outputDirectory "models"
New-Item -ItemType Directory -Path $modelsOutput -Force | Out-Null
foreach ($filename in $modelFiles) {
    Copy-Item -LiteralPath (Join-Path $ModelDirectory $filename) -Destination (Join-Path $modelsOutput $filename) -Force
}

$executable = Join-Path $outputDirectory "jade_pose_server.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "The pose server executable was not produced: $executable"
}
Write-Host "Bundled pose server: $executable"
