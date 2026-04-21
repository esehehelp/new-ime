param(
    [string]$DllPath = "$PSScriptRoot\..\..\build\release\new_ime_tsf.dll"
)

$resolved = (Resolve-Path $DllPath).Path
$escaped = $resolved.Replace('\', '\\')

$source = @"
using System;
using System.Runtime.InteropServices;
public static class NewImeTsfRsUnregistration {
    [DllImport("$escaped", ExactSpelling=true)]
    public static extern int DllUnregisterServer();
}
"@

Add-Type -TypeDefinition $source
$hr = [NewImeTsfRsUnregistration]::DllUnregisterServer()
if ($hr -ne 0) {
    throw ("DllUnregisterServer failed: 0x{0:X8}" -f $hr)
}

Write-Host "Unregistered Rust TSF DLL:" $resolved
