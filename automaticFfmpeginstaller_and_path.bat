@echo off
setlocal enabledelayedexpansion

echo FFmpeg Windows Installer and PATH Setup
echo =======================================
echo.

:: Check for admin privileges
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if %errorlevel% neq 0 (
    echo Administrator privileges required!
    echo Please right-click this batch file and select "Run as administrator"
    echo.
    pause
    exit /B 1
)

:: Set installation directory
set "INSTALL_DIR=C:\ffmpeg"
echo FFmpeg will be installed to: %INSTALL_DIR%
echo.

:: Create directory if it doesn't exist
if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
    if %errorlevel% neq 0 (
        echo Failed to create installation directory.
        pause
        exit /B 1
    )
)

:: Download FFmpeg
echo Downloading FFmpeg... (This might take a while)
echo.

:: Use PowerShell to download the file
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile '%TEMP%\ffmpeg.zip'}"

if %errorlevel% neq 0 (
    echo Failed to download FFmpeg.
    pause
    exit /B 1
)

echo Download complete!
echo.

:: Extract the ZIP file
echo Extracting FFmpeg...
echo.

powershell -Command "& {Expand-Archive -Path '%TEMP%\ffmpeg.zip' -DestinationPath '%TEMP%\ffmpeg_extract' -Force}"

if %errorlevel% neq 0 (
    echo Failed to extract FFmpeg.
    pause
    exit /B 1
)

:: Move contents to installation directory
echo Moving files to installation directory...
echo.

:: Find the extracted folder (it usually has a version in the name)
for /d %%G in ("%TEMP%\ffmpeg_extract\ffmpeg*") do (
    xcopy "%%G\bin\*" "%INSTALL_DIR%\bin\" /E /I /Y
    if !errorlevel! neq 0 (
        echo Failed to copy FFmpeg files.
        pause
        exit /B 1
    )
)

echo Files moved successfully.
echo.

:: Add to PATH
echo Adding FFmpeg to PATH...
echo.

:: Check if already in PATH
echo %PATH% | findstr /C:"%INSTALL_DIR%\bin" >nul
if %errorlevel% equ 0 (
    echo FFmpeg is already in your PATH.
) else (
    :: Add to PATH using setx
    setx PATH "%PATH%;%INSTALL_DIR%\bin" /M
    if %errorlevel% neq 0 (
        echo Failed to add FFmpeg to PATH.
        pause
        exit /B 1
    )
    echo FFmpeg added to PATH successfully!
)

echo.
echo Cleaning up temporary files...
del "%TEMP%\ffmpeg.zip"
rmdir /S /Q "%TEMP%\ffmpeg_extract"

echo.
echo =======================================
echo Installation complete!
echo.
echo Please close and reopen any command prompt windows for the PATH changes to take effect.
echo You can run "ffmpeg -version" to verify the installation.
echo.
pause
