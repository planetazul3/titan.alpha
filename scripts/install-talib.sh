#!/bin/bash
# TA-Lib Installation Script
# Run this BEFORE pip install TA-Lib
#
# Usage: bash scripts/install-talib.sh
# For Colab: !bash scripts/install-talib.sh

set -e

echo "=== Installing TA-Lib System Library ==="

# Check if already installed
if command -v ta-lib-config &> /dev/null; then
    echo "TA-Lib is already installed."
    ta-lib-config --version
    exit 0
fi

# For Debian/Ubuntu/Colab
if command -v apt-get &> /dev/null; then
    echo "Detected apt-get (Debian/Ubuntu/Colab)"
    
    # Try installing from package manager first
    if apt-cache show ta-lib &> /dev/null 2>&1 || apt-cache show libta-lib-dev &> /dev/null 2>&1; then
        echo "Installing from package manager..."
        sudo apt-get update
        sudo apt-get install -y ta-lib libta-lib-dev 2>/dev/null || sudo apt-get install -y libta-lib0 libta-lib-dev
    else
        # Build from source
        echo "Package not available, building from source..."
        sudo apt-get update
        sudo apt-get install -y build-essential wget
        
        cd /tmp
        wget -q https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        
        # M18: Verify checksum
        EXPECTED_HASH="9ff41efcb1c011a4b4b6dfc91610b06e39b1d7973ed5d4dee55029a0ac4dc651"
        FILE_HASH=$(sha256sum ta-lib-0.4.0-src.tar.gz | cut -d' ' -f1)
        
        if [ "$FILE_HASH" != "$EXPECTED_HASH" ]; then
            echo "ERROR: Checksum mismatch!"
            echo "Expected: $EXPECTED_HASH"
            echo "Got:      $FILE_HASH"
            rm ta-lib-0.4.0-src.tar.gz
            exit 1
        fi
        echo "Checksum verified."

        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr
        make
        sudo make install
        cd ..
        rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    fi
    
# For RHEL/CentOS/Fedora
elif command -v yum &> /dev/null; then
    echo "Detected yum (RHEL/CentOS)"
    sudo yum install -y ta-lib ta-lib-devel

elif command -v dnf &> /dev/null; then
    echo "Detected dnf (Fedora)"
    sudo dnf install -y ta-lib ta-lib-devel

# For macOS
elif command -v brew &> /dev/null; then
    echo "Detected Homebrew (macOS)"
    brew install ta-lib

else
    echo "ERROR: Could not detect package manager."
    echo "Please install TA-Lib manually from: http://ta-lib.org/"
    exit 1
fi

echo ""
echo "=== TA-Lib System Library Installed Successfully ==="
echo ""
echo "Now you can install the Python wrapper:"
echo "  pip install TA-Lib"
