@echo off
setlocal enabledelayedexpansion
REM Build new-ime-engine.dll + test_ffi.exe into build/win32
REM
REM DLL backend:
REM   - ONNX Runtime CTC-NAT inference
REM   - CTC prefix beam search
REM   - KenLM shallow fusion

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..") do set REPO_ROOT=%%~fI
set OUT_DIR=%REPO_ROOT%\build\win32

set ORT_DIR=%REPO_ROOT%\tools\onnxruntime-win-x64-1.22.0
set KENLM_SRC=%REPO_ROOT%\engine\server\third_party\kenlm
set KENLM_BUILD=%REPO_ROOT%\build\kenlm_win
set SERVER_SRC=%REPO_ROOT%\engine\server\src

if not exist "%ORT_DIR%" (
    echo onnxruntime not found at %ORT_DIR%
    echo Extract onnxruntime-win-x64-1.22.0.zip under tools/ and retry.
    exit /b 1
)
if not exist "%KENLM_SRC%" (
    echo kenlm source not found at %KENLM_SRC%
    echo git clone https://github.com/kpu/kenlm engine/server/third_party/kenlm
    exit /b 1
)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%KENLM_BUILD%\lib\Release\kenlm.lib" (
    echo === Building KenLM ===
    if not exist "%KENLM_BUILD%" mkdir "%KENLM_BUILD%"
    pushd "%KENLM_BUILD%"
    cmake -DCMAKE_BUILD_TYPE=Release ^
          -DFORCE_STATIC=ON -DENABLE_PYTHON=OFF -DKENLM_MAX_ORDER=6 ^
          -DBUILD_TESTING=OFF -DKENLM_LIBS_ONLY=ON ^
          "%KENLM_SRC%"
    if !ERRORLEVEL! NEQ 0 (
        echo CMake configure for KenLM failed
        popd
        exit /b 1
    )
    cmake --build . --config Release --target kenlm kenlm_util
    if !ERRORLEVEL! NEQ 0 (
        echo CMake build for KenLM failed
        popd
        exit /b 1
    )
    popd
) else (
    echo KenLM already built, skipping
)

pushd "%SCRIPT_DIR%"

echo === Building DLL ===
cl /LD /EHsc /std:c++20 /O2 /utf-8 ^
    /DNEWIME_ENABLE_KENLM /DNOMINMAX /DKENLM_MAX_ORDER=6 ^
    /I"%ORT_DIR%\include" ^
    /I"%SERVER_SRC%" ^
    /I"%KENLM_SRC%" ^
    "%SERVER_SRC%\ctc_decoder.cpp" ^
    "%SERVER_SRC%\lm_scorer_kenlm.cpp" ^
    ffi_impl.cpp ^
    /Fo:"%OUT_DIR%\\" ^
    /Fe:"%OUT_DIR%\new-ime-engine.dll" ^
    /link ^
    /MAP:"%OUT_DIR%\new-ime-engine.map" ^
    /LIBPATH:"%ORT_DIR%\lib" ^
    /LIBPATH:"%KENLM_BUILD%\lib\Release" ^
    /LIBPATH:"%KENLM_BUILD%\lib" ^
    onnxruntime.lib ^
    kenlm.lib kenlm_util.lib

if %ERRORLEVEL% NEQ 0 (
    echo DLL BUILD FAILED
    exit /b 1
)
echo DLL OK

echo === Building test ===
cl /EHsc /std:c++20 /O2 /utf-8 ^
    test_ffi.cpp ^
    /Fo:"%OUT_DIR%\\" ^
    /Fe:"%OUT_DIR%\test_ffi.exe" ^
    /link

if %ERRORLEVEL% NEQ 0 (
    echo TEST BUILD FAILED
    exit /b 1
)
echo TEST OK

echo === Copying ONNX Runtime DLL ===
copy "%ORT_DIR%\lib\onnxruntime.dll" "%OUT_DIR%\" >nul 2>&1
copy "%ORT_DIR%\lib\onnxruntime_providers_shared.dll" "%OUT_DIR%\" >nul 2>&1

echo === Building interactive demo ===
cl /EHsc /std:c++20 /O2 /utf-8 ^
    interactive.cpp ^
    /Fo:"%OUT_DIR%\\" ^
    /Fe:"%OUT_DIR%\interactive.exe" ^
    /link

popd
echo === Ready ===
echo Output: %OUT_DIR%
echo Run: %OUT_DIR%\test_ffi.exe (automated test)
echo Run: %OUT_DIR%\interactive.exe (interactive demo)
