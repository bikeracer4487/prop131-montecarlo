#!/bin/bash
# === Build Script for MacOS ===

echo "Cleaning previous build directories..."

# Remove build directory if it exists
if [ -d "build" ]; then
    rm -rf build
    echo "Removed 'build' directory."
else
    echo "'build' directory does not exist. Skipping."
fi

# Remove dist directory if it exists
if [ -d "dist" ]; then
    rm -rf dist
    echo "Removed 'dist' directory."
else
    echo "'dist' directory does not exist. Skipping."
fi

# Remove spec file if it exists
if [ -f "ElectionSimulator.spec" ]; then
    rm ElectionSimulator.spec
    echo "Removed 'ElectionSimulator.spec' file."
else
    echo "'ElectionSimulator.spec' does not exist. Skipping."
fi

echo "Starting the build process..."

pyinstaller --onedir --windowed --name "ElectionSimulator" --icon=icon.icns --add-data "icon.png:." election_simulator.py

echo "Build process completed."
