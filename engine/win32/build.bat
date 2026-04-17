@echo off
REM Build new-ime-engine.dll + test_ffi.exe

set ORT_DIR=C:\Users\admin\Dev\new-ime\tools\onnxruntime-win-x64-1.22.0

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

echo === Building DLL ===
cl /LD /EHsc /std:c++17 /O2 /utf-8 ^
    /I"%ORT_DIR%\include" ^
    ffi_impl.cpp ^
    /Fe:new-ime-engine.dll ^
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
    /Fe:test_ffi.exe ^
    /link

if %ERRORLEVEL% NEQ 0 (
    echo TEST BUILD FAILED
    exit /b 1
)
echo TEST OK

echo === Copying ONNX Runtime DLL ===
copy "%ORT_DIR%\lib\onnxruntime.dll" . >nul 2>&1
copy "%ORT_DIR%\lib\onnxruntime_providers_shared.dll" . >nul 2>&1

echo === Ready ===
echo Run: test_ffi.exe
