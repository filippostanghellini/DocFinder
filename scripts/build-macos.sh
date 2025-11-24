#!/bin/bash
# Build script for macOS - creates DocFinder.app and DocFinder.dmg
#
# Prerequisites:
#   brew install create-dmg
#   pip install '.[dev,gui]'
#
# Usage:
#   ./scripts/build-macos.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="$PROJECT_ROOT/resources"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=== DocFinder macOS Build Script ==="
echo "Project root: $PROJECT_ROOT"

# Create resources directory if it doesn't exist
mkdir -p "$RESOURCES_DIR"

# Convert Logo.png to .icns if needed
LOGO_PNG="$PROJECT_ROOT/Logo.png"
ICNS_FILE="$RESOURCES_DIR/DocFinder.icns"

if [ -f "$LOGO_PNG" ]; then
    echo "Converting Logo.png to .icns..."
    
    # Create iconset directory
    ICONSET_DIR="$RESOURCES_DIR/DocFinder.iconset"
    mkdir -p "$ICONSET_DIR"
    
    # Generate all required icon sizes
    sips -z 16 16     "$LOGO_PNG" --out "$ICONSET_DIR/icon_16x16.png" 2>/dev/null
    sips -z 32 32     "$LOGO_PNG" --out "$ICONSET_DIR/icon_16x16@2x.png" 2>/dev/null
    sips -z 32 32     "$LOGO_PNG" --out "$ICONSET_DIR/icon_32x32.png" 2>/dev/null
    sips -z 64 64     "$LOGO_PNG" --out "$ICONSET_DIR/icon_32x32@2x.png" 2>/dev/null
    sips -z 128 128   "$LOGO_PNG" --out "$ICONSET_DIR/icon_128x128.png" 2>/dev/null
    sips -z 256 256   "$LOGO_PNG" --out "$ICONSET_DIR/icon_128x128@2x.png" 2>/dev/null
    sips -z 256 256   "$LOGO_PNG" --out "$ICONSET_DIR/icon_256x256.png" 2>/dev/null
    sips -z 512 512   "$LOGO_PNG" --out "$ICONSET_DIR/icon_256x256@2x.png" 2>/dev/null
    sips -z 512 512   "$LOGO_PNG" --out "$ICONSET_DIR/icon_512x512.png" 2>/dev/null
    sips -z 1024 1024 "$LOGO_PNG" --out "$ICONSET_DIR/icon_512x512@2x.png" 2>/dev/null
    
    # Convert iconset to icns
    iconutil -c icns "$ICONSET_DIR" -o "$ICNS_FILE"
    
    # Clean up iconset
    rm -rf "$ICONSET_DIR"
    
    echo "Created $ICNS_FILE"
else
    echo "Warning: Logo.png not found, building without custom icon"
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf "$DIST_DIR/DocFinder"
rm -rf "$DIST_DIR/DocFinder.app"
rm -f "$DIST_DIR/DocFinder.dmg"
rm -rf "$BUILD_DIR/DocFinder"

# Run PyInstaller
echo "Running PyInstaller..."
cd "$PROJECT_ROOT"

# Use pyinstaller from venv if available, otherwise system
if [ -x "$PROJECT_ROOT/.venv/bin/pyinstaller" ]; then
    PYINSTALLER="$PROJECT_ROOT/.venv/bin/pyinstaller"
else
    PYINSTALLER="pyinstaller"
fi

$PYINSTALLER DocFinder.spec --clean --noconfirm

# Verify the app was created
if [ ! -d "$DIST_DIR/DocFinder.app" ]; then
    echo "Error: DocFinder.app was not created"
    exit 1
fi

echo "DocFinder.app created successfully"

# Create DMG
echo "Creating DMG installer..."

# Check if create-dmg is installed
if ! command -v create-dmg &> /dev/null; then
    echo "Warning: create-dmg not installed. Install with: brew install create-dmg"
    echo "Skipping DMG creation. You can find the app at: $DIST_DIR/DocFinder.app"
    exit 0
fi

# Remove existing DMG if present
rm -f "$DIST_DIR/DocFinder.dmg"

# Create DMG with create-dmg
create-dmg \
    --volname "DocFinder" \
    --volicon "$ICNS_FILE" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "DocFinder.app" 150 190 \
    --hide-extension "DocFinder.app" \
    --app-drop-link 450 190 \
    --no-internet-enable \
    "$DIST_DIR/DocFinder.dmg" \
    "$DIST_DIR/DocFinder.app"

echo ""
echo "=== Build Complete ==="
echo "App:       $DIST_DIR/DocFinder.app"
echo "Installer: $DIST_DIR/DocFinder.dmg"
echo ""
echo "To install:"
echo "  1. Open DocFinder.dmg"
echo "  2. Drag DocFinder.app to Applications"
echo "  3. On first launch, right-click > Open to bypass Gatekeeper"
