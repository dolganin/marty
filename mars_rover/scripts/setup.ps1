param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$BuildType = "Debug",
    [ValidateSet("Auto", "MSVC", "GCC")]
    [string]$Toolchain = "Auto",
    [string]$Python = "python",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv"

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    Remove-Item -LiteralPath $Venv -Recurse -Force
}
if (-not (Test-Path -LiteralPath $Venv)) {
    Invoke-Checked { & $Python -m venv $Venv }
}
$VenvPython = Join-Path $Venv "Scripts\python.exe"
Invoke-Checked { & $VenvPython -m pip install --upgrade pip setuptools wheel }
Invoke-Checked { & $VenvPython -m pip install pybind11 }
$env:MARS_ROVER_BUILD_TYPE = $BuildType
if ($Toolchain -eq "Auto") {
    $Toolchain = if (Get-Command cl.exe -ErrorAction SilentlyContinue) { "MSVC" } elseif (Get-Command g++.exe -ErrorAction SilentlyContinue) { "GCC" } else { "MSVC" }
}

if ($Toolchain -eq "GCC") {
    $env:MARS_ROVER_SKIP_NATIVE = "1"
    Invoke-Checked { & $VenvPython -m pip install --editable "$ProjectRoot[dev]" --no-build-isolation }
    Remove-Item Env:MARS_ROVER_SKIP_NATIVE -ErrorAction SilentlyContinue
    $BuildDir = Join-Path $ProjectRoot "build-python-gcc"
    $PythonCMakePath = $VenvPython.Replace("\", "/")
    Invoke-Checked {
        & cmake -S (Join-Path $ProjectRoot "cpp") -B $BuildDir -G Ninja `
            "-DPython_EXECUTABLE=$PythonCMakePath" -DMARS_ROVER_BUILD_PYTHON=ON `
            "-DCMAKE_BUILD_TYPE=$BuildType"
    }
    Invoke-Checked { & cmake --build $BuildDir --parallel }
    $SitePackages = & $VenvPython -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    if ($LASTEXITCODE -ne 0) { throw "Could not locate site-packages" }
    Copy-Item (Join-Path $BuildDir "_mars_rover_cpp*.pyd") $SitePackages -Force
    Copy-Item (Join-Path $BuildDir "_mars_rover_cpp*.pyd") (Join-Path $ProjectRoot "python") -Force
    $DllDir = Join-Path $SitePackages "_mars_rover_dlls"
    New-Item -ItemType Directory -Force -Path $DllDir | Out-Null
    $Gxx = (Get-Command g++.exe -ErrorAction Stop).Source
    $GccBin = Split-Path -Parent $Gxx
    foreach ($Dll in "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll", "libgomp-1.dll", "libdl.dll") {
        $DllPath = Join-Path $GccBin $Dll
        if (Test-Path -LiteralPath $DllPath) { Copy-Item $DllPath $DllDir -Force }
    }
} else {
    Invoke-Checked { & $VenvPython -m pip install --editable "$ProjectRoot[dev]" --no-build-isolation }
}
Invoke-Checked { & $VenvPython -m mars_rover_env.tools.doctor }
Write-Host "Ready ($Toolchain/$BuildType). Play: .\.venv\Scripts\mars-rover-play.exe --debug"
