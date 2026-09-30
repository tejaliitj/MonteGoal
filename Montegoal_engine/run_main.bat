@echo off
setlocal
cd /d "%~dp0"

REM Optional first argument: philox | xorwow | mrg32k3a  (default: philox)
set "RNG=philox"
set "RNGFLAG=-DRNG_PHILOX"
set "PICKED="
if /i "%~1"=="philox"   set "PICKED=1"
if /i "%~1"=="xorwow"   (set "RNG=xorwow"   & set "RNGFLAG=-DRNG_XORWOW"   & set "PICKED=1")
if /i "%~1"=="mrg32k3a" (set "RNG=mrg32k3a" & set "RNGFLAG=-DRNG_MRG32K3A" & set "PICKED=1")
if defined PICKED shift

REM Remaining arguments = trials, seed, --block N
set "ARGS="
:collect
if "%~1"=="" goto collected
set "ARGS=%ARGS% %1"
shift
goto collect
:collected

if not exist build mkdir build
where nvcc >nul 2>nul
if errorlevel 1 (
    echo ERROR: nvcc was not found on PATH.
    pause
    exit /b 1
)

nvcc -std=c++17 -O3 -lineinfo -rdc=true -arch=native ^
-Iengine -Irng -Ifootball ^
%RNGFLAG% ^
main.cu ^
football/tournament_data.cu ^
football/third_place_data.cu ^
football/bracket_data.cu ^
football/lambda_matrix.cu ^
-o build\montegoal_%RNG%.exe
if errorlevel 1 (
    echo BUILD FAILED.
    pause
    exit /b 1
)

build\montegoal_%RNG%.exe %ARGS%
if errorlevel 1 (
    echo PROGRAM FAILED.
    pause
    exit /b 1
)

pause
endlocal