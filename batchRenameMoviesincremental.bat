@echo off
setlocal enabledelayedexpansion

echo MP4 Renaming Tool
echo -----------------
echo This script will rename all MP4 files in the current folder to sequential numbers (01.mp4, 02.mp4, etc.)
echo.

:: Count the number of MP4 files
set count=0
for %%f in (*.mp4) do set /a count+=1

echo Found %count% MP4 files to rename.
echo.

:: Confirm with user
set /p confirm=Are you sure you want to rename these files? (Y/N): 
if /i not "%confirm%"=="Y" goto :end

:: Create a temporary directory for the process
if not exist "temp_rename" mkdir temp_rename

:: First move all files to temp directory with numbered names
set filenum=0
for %%f in (*.mp4) do (
    set /a filenum+=1
    
    :: Format the number with leading zero if needed
    if !filenum! LSS 10 (
        set "padded_num=0!filenum!"
    ) else (
        set "padded_num=!filenum!"
    )
    
    echo Moving: "%%f" to temp_rename\!padded_num!.mp4
    copy "%%f" "temp_rename\!padded_num!.mp4" > nul
)

:: Delete original files
for %%f in (*.mp4) do (
    del "%%f"
)

:: Move renamed files back to original directory
move "temp_rename\*.mp4" "." > nul

:: Remove temporary directory
rmdir "temp_rename"

echo.
echo Renaming complete! Files are now numbered from 01.mp4 to %padded_num%.mp4

:end
echo.
pause
