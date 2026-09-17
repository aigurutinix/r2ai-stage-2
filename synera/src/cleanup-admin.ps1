$ErrorActionPreference = 'Stop'
Start-Transcript -Path 'E:\Road2AI\cleanup-admin-result.txt' -Force
try {
    $key = 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\gxx'
    if (Test-Path -LiteralPath $key) {
        if ((Get-ItemProperty -LiteralPath $key).DisplayName -ne 'Garena (remove only)') { throw 'Unexpected registry target' }
        if (Test-Path -LiteralPath 'C:\Program Files (x86)\Garena\Garena\2.0.1909.2618\uninst.exe') { throw 'Uninstaller exists' }
        Remove-Item -LiteralPath $key -Recurse -Force
        Write-Output 'Removed stale Garena registration'
    }
    diskpart /s E:\Road2AI\compact-ubuntu-cleanup.txt
    Get-Item -LiteralPath 'C:\Users\DELL\AppData\Local\wsl\{e9cbbf70-7588-4c31-bca3-c4598f6bb64c}\ext4.vhdx' | Select-Object Length | Format-List
    Get-PSDrive C | Select-Object Free | Format-List
} finally { Stop-Transcript }
