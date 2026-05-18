@echo off
echo Starting needlestack installer...
echo.
echo If Windows asks "Do you want to allow this app to make changes",
echo click Yes -- it is needed to install the software.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-windows.ps1"
echo.
pause
