@echo off
setlocal
REM Build new-ime-engine.dll + test_ffi.exe into build/win32

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..") do set REPO_ROOT=%%~fI
set OUT_DIR=%REPO_ROOT%\build\win32

set ORT_DIR=C:\Users\admin\Dev\new-ime\tools\onnxruntime-win-x64-1.22.0

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
pushd "%SCRIPT_DIR%"

echo === Building DLL ===
cl /LD /EHsc /std:c++17 /O2 /utf-8 ^
    /I"%ORT_DIR%\include" ^
    ffi_impl.cpp ^
    /Fo:"%OUT_DIR%\\" ^
    /Fe:"%OUT_DIR%\new-ime-engine.dll" ^
    /link ^
    /LIBPATH:"%ORT_DIR%\lib" ^
    onnxruntime.lib

if %ERRORLEVEL% NEQ 0 (
    echo DLL BUILD FAILED
    exit /b 1
)
echo DLL OK

echo === Building test ===
cl /EHsc /std:c++17 /O2 /utf-8 ^
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
cl /EHsc /std:c++17 /O2 /utf-8 ^
    interactive.cpp ^
    /Fo:"%OUT_DIR%\\" ^
    /Fe:"%OUT_DIR%\interactive.exe" ^
    /link

popd
echo === Ready ===
echo Output: %OUT_DIR%
echo Run: %OUT_DIR%\test_ffi.exe (automated test)
echo Run: %OUT_DIR%\interactive.exe (interactive demo)
