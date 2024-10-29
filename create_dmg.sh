#!/bin/bash
# === DMG Creation Script for MacOS ===

# Set variables
APP_NAME="ElectionSimulator.app"
DMG_NAME="ElectionSimulator.dmg"
TEMP_DIR="dmg_source"
ICON_PATH="icon.png"

# Paths
APP_PATH="dist/$APP_NAME"
TEMP_APP_PATH="$TEMP_DIR/$APP_NAME"

# Check if the .app exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: '$APP_PATH' does not exist."
    echo "Please build the application first using build_macos.sh."
    exit 1
fi

# Check if the icon file exists
if [ ! -f "$ICON_PATH" ]; then
    echo "Error: '$ICON_PATH' does not exist."
    echo "Please ensure 'icon.png' is present."
    exit 1
fi

# Clean up previous temporary directory
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Create temporary directory and copy the .app into it
mkdir "$TEMP_DIR"
cp -R "$APP_PATH" "$TEMP_DIR/"

# Remove old DMG if it exists
if [ -f "$DMG_NAME" ]; then
    rm "$DMG_NAME"
    echo "Removed old '$DMG_NAME' file."
else
    echo "'$DMG_NAME' does not exist. Skipping removal."
fi

echo "Creating DMG..."

create-dmg \
  --volname "ElectionSimulator" \
  --volicon "icon.icns" \
  --codesign "Developer ID Application: Douglas Mason (3PAP25XR7L)" \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "$APP_NAME" 150 120 \
  --app-drop-link 450 120 \
  "$DMG_NAME" \
  "$TEMP_DIR/"

echo "DMG '$DMG_NAME' created successfully."

# Clean up temporary directory
rm -rf "$TEMP_DIR"
