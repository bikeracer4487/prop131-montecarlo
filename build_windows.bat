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
if exist ElectionSimulator.spec (
    del /f /q ElectionSimulator.spec
    echo Removed 'ElectionSimulator.spec' file.
) else (
    echo 'ElectionSimulator.spec' does not exist. Skipping.
)

echo Starting the build process...

pyinstaller --onedir --windowed --name "ElectionSimulator" --icon=icon.ico --add-data "icon.ico;." election_simulator.py

echo Build process completed.

pause
