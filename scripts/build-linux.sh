#!/bin/bash
# Build script for Linux - creates DocFinder AppImage
#
# Prerequisites:
#   - pip install '.[dev,gui]'
#   - wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
#   - chmod +x appimagetool-x86_64.AppImage
#
# Usage:
#   ./scripts/build-linux.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="$PROJECT_ROOT/resources"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"
APPDIR="$DIST_DIR/DocFinder.AppDir"

echo "=== DocFinder Linux Build Script ==="
echo "Project root: $PROJECT_ROOT"

# Create resources directory if it doesn't exist
mkdir -p "$RESOURCES_DIR"

# Copy Logo.png to resources if it exists
LOGO_PNG="$PROJECT_ROOT/Logo.png"
if [ -f "$LOGO_PNG" ]; then
    cp "$LOGO_PNG" "$RESOURCES_DIR/docfinder.png"
    echo "Copied Logo.png to resources"
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf "$DIST_DIR/DocFinder"
rm -rf "$APPDIR"
rm -f "$DIST_DIR/DocFinder.AppImage"
rm -rf "$BUILD_DIR/DocFinder"

# Run PyInstaller
echo "Running PyInstaller..."
cd "$PROJECT_ROOT"
pyinstaller DocFinder.spec --clean --noconfirm

# Verify the build was created
if [ ! -d "$DIST_DIR/DocFinder" ]; then
    echo "Error: DocFinder build was not created"
    exit 1
fi

echo "DocFinder built successfully"

# Create AppImage structure
echo "Creating AppImage structure..."
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$APPDIR/usr/share/metainfo"

# Copy the built application
cp -r "$DIST_DIR/DocFinder/"* "$APPDIR/usr/bin/"

# Create AppRun script
cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/DocFinder" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# Copy icon
if [ -f "$RESOURCES_DIR/docfinder.png" ]; then
    cp "$RESOURCES_DIR/docfinder.png" "$APPDIR/docfinder.png"
    cp "$RESOURCES_DIR/docfinder.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/docfinder.png"
    cp "$RESOURCES_DIR/docfinder.png" "$APPDIR/.DirIcon"
fi

# Create .desktop file
cat > "$APPDIR/docfinder.desktop" << EOF
[Desktop Entry]
Type=Application
Name=DocFinder
Comment=Local semantic search for PDF documents
Exec=DocFinder
Icon=docfinder
Categories=Office;Utility;
Terminal=false
StartupNotify=true
EOF

# Also copy to standard location
cp "$APPDIR/docfinder.desktop" "$APPDIR/usr/share/applications/"

# Create AppStream metainfo
cat > "$APPDIR/usr/share/metainfo/com.docfinder.app.metainfo.xml" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
  <id>com.docfinder.app</id>
  <name>DocFinder</name>
  <summary>Local semantic search for PDF documents</summary>
  <metadata_license>MIT</metadata_license>
  <project_license>MIT</project_license>
  <description>
    <p>
      DocFinder is a local-first semantic search tool for PDF documents.
      It uses sentence-transformers to create embeddings and allows you to
      search your documents using natural language queries.
    </p>
    <p>Features:</p>
    <ul>
      <li>Extract text from PDFs with configurable chunking</li>
      <li>Generate local embeddings using ONNX for fast inference</li>
      <li>Perform semantic search across all indexed documents</li>
      <li>No cloud services required - everything runs locally</li>
    </ul>
  </description>
  <url type="homepage">https://github.com/filippostanghellini/DocFinder</url>
  <url type="bugtracker">https://github.com/filippostanghellini/DocFinder/issues</url>
  <launchable type="desktop-id">docfinder.desktop</launchable>
  <content_rating type="oars-1.1"/>
  <releases>
    <release version="0.1.0" date="2025-11-08"/>
  </releases>
</component>
EOF

echo "AppImage structure created"

# Try to create AppImage
APPIMAGETOOL="$PROJECT_ROOT/appimagetool-x86_64.AppImage"
if [ ! -f "$APPIMAGETOOL" ]; then
    APPIMAGETOOL="$(which appimagetool 2>/dev/null || true)"
fi

if [ -x "$APPIMAGETOOL" ]; then
    echo "Creating AppImage..."
    ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$DIST_DIR/DocFinder-x86_64.AppImage"
    
    echo ""
    echo "=== Build Complete ==="
    echo "AppImage: $DIST_DIR/DocFinder-x86_64.AppImage"
    echo ""
    echo "To run:"
    echo "  chmod +x DocFinder-x86_64.AppImage"
    echo "  ./DocFinder-x86_64.AppImage"
else
    echo ""
    echo "Warning: appimagetool not found."
    echo "Download from: https://github.com/AppImage/AppImageKit/releases"
    echo ""
    echo "The AppDir structure is ready at: $APPDIR"
    echo "You can run the app directly with: $APPDIR/AppRun"
fi
