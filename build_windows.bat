@echo off
REM === Build Script for Windows ===

echo Cleaning previous build directories...

REM Remove build directory if it exists
if exist build (
    rd /s /q build
    echo Removed 'build' directory.
) else (
    echo 'build' directory does not exist. Skipping.
)

REM Remove dist directory if it exists
if exist dist (
    rd /s /q dist
    echo Removed 'dist' directory.
) else (
    echo 'dist' directory does not exist. Skipping.
)

REM Remove spec file if it exists
if exist AnalyticsParser.spec (
    del /f /q AnalyticsParser.spec
    echo Removed 'AnalyticsParser.spec' file.
) else (
    echo 'AnalyticsParser.spec' does not exist. Skipping.
)

echo Starting the build process...

pyinstaller --onefile --windowed --name "AnalyticsParser" --icon=icon.ico --add-data "icon.ico;." analytics_parser.py

echo Build process completed.

pause
