# Build script for Windows using nvcc and MSVC tools
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    $found = Get-ChildItem "C:\Program Files\Microsoft Visual Studio", "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $vcvars = $found.FullName }
}

Write-Host "Compiling Monte Carlo Engine (DLL and EXE)..." -ForegroundColor Cyan
cmd /c "`"$vcvars`" && nvcc -O3 -std=c++17 -arch=sm_86 --shared -Xcompiler `"/MD`" -lcurand src/main.cu -o montecarlo_engine.dll && nvcc -O3 -std=c++17 -arch=sm_86 -Xcompiler `"/MD`" -lcurand src/main.cu -o montecarlo_engine.exe"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build Succeeded: montecarlo_engine.dll & montecarlo_engine.exe" -ForegroundColor Green
} else {
    Write-Host "Build Failed!" -ForegroundColor Red
}
