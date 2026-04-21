param(
    [string]$DllPath = "$PSScriptRoot\..\..\build\release\new_ime_tsf.dll"
)

# Register the Rust-native new-ime TSF DLL.
# Must be run in a PowerShell session with the permissions needed to write
# HKCU\Software\Classes\CLSID (non-admin is fine on Windows 10/11).

$resolved = (Resolve-Path $DllPath).Path
$escaped = $resolved.Replace('\', '\\')

$source = @"
using System;
using System.Runtime.InteropServices;
public static class NewImeTsfRsRegistration {
    [DllImport("$escaped", ExactSpelling=true)]
    public static extern int DllRegisterServer();
}
"@

Add-Type -TypeDefinition $source
$hr = [NewImeTsfRsRegistration]::DllRegisterServer()
if ($hr -ne 0) {
    throw ("DllRegisterServer failed: 0x{0:X8}" -f $hr)
}

Write-Host "Registered Rust TSF DLL:" $resolved
Write-Host "Sign out and back in (or reboot) to pick up the registration in all hosts."
