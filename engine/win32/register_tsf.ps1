param(
    [string]$DllPath = "$PSScriptRoot\..\..\build\win32\new-ime-tsf.dll"
)

$resolved = (Resolve-Path $DllPath).Path
$escaped = $resolved.Replace('\', '\\')

$source = @"
using System;
using System.Runtime.InteropServices;
public static class NewImeTsfRegistration {
    [DllImport("$escaped", ExactSpelling=true)]
    public static extern int DllRegisterServer();
}
"@

Add-Type -TypeDefinition $source
$hr = [NewImeTsfRegistration]::DllRegisterServer()
if ($hr -ne 0) {
    throw ("DllRegisterServer failed: 0x{0:X8}" -f $hr)
}

Write-Host "Registered TSF DLL:" $resolved
