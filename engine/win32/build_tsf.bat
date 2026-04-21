@echo off
setlocal enabledelayedexpansion
REM Build new-ime-tsf.dll into build/win32

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..") do set REPO_ROOT=%%~fI
set OUT_DIR=%REPO_ROOT%\build\win32

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
pushd "%SCRIPT_DIR%"

echo === Building TSF DLL ===
cl /LD /EHsc /std:c++20 /O2 /utf-8 ^
    /DUNICODE /D_UNICODE /DWIN32_LEAN_AND_MEAN /DNOMINMAX ^
    /I"%REPO_ROOT%\engine\src" ^
    tsf_guids.cpp ^
    tsf_backend.cpp ^
    tsf_text_service.cpp ^
    "%REPO_ROOT%\engine\src\composing_text.cpp" ^
    /Fo:"%OUT_DIR%\\" ^
    /Fe:"%OUT_DIR%\new-ime-tsf.dll" ^
    /link ^
    /DEF:new-ime-tsf.def ^
    ole32.lib uuid.lib advapi32.lib user32.lib

if %ERRORLEVEL% NEQ 0 (
    echo TSF BUILD FAILED
    exit /b 1
)

popd
echo === TSF READY ===
echo Output: %OUT_DIR%\new-ime-tsf.dll

echo === Building TSF enum tool ===
cl /EHsc /std:c++20 /O2 /utf-8 ^
    /DUNICODE /D_UNICODE /DWIN32_LEAN_AND_MEAN /DNOMINMAX ^
    "%SCRIPT_DIR%tsf_guids.cpp" ^
    "%SCRIPT_DIR%tsf_enum.cpp" ^
    /Fo:"%OUT_DIR%\\" ^
    /Fe:"%OUT_DIR%\tsf_enum.exe" ^
    /link ^
    ole32.lib uuid.lib advapi32.lib user32.lib

if %ERRORLEVEL% NEQ 0 (
    echo TSF ENUM BUILD FAILED
    exit /b 1
)

echo Output: %OUT_DIR%\tsf_enum.exe
