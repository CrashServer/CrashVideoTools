@echo off
setlocal enabledelayedexpansion
:: Use current directory for input
set "input_dir=%cd%"
:: Create output directory as a subfolder of current directory
set "output_dir=%cd%\converted"
:: Create output directory if it doesn't exist
if not exist "%output_dir%" mkdir "%output_dir%"
echo Input directory: %input_dir%
echo Output directory: %output_dir%
:: Process each MP4 file in the current directory
for %%f in ("*.mp4") do (
    :: Skip already converted files if they exist in the output folder
    if not "%%~nxf"=="converted\%%~nxf" (
        echo Processing: %%~nxf
        ffmpeg -i "%%f" -an -movflags +faststart -brand mp42 -pix_fmt yuv420p -profile:v baseline -level 3.0 "%output_dir%\%%~nxf"
        echo Completed: %%~nxf
    )
)

echo All videos processed!
