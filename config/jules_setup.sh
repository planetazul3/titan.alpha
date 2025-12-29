#!/bin/bash
# =============================================================================
# JULES INITIAL SETUP SCRIPT (FORK STRATEGY)
# Copy content of this file to: 
# Codebases -> x.titan -> Configuration -> Initial Setup
# =============================================================================
set -e

echo "🔧 Configuring environment for x.titan (Fork Strategy)..."

# 0. Save Repo Root
REPO_ROOT=$(pwd)

# 1. Install TA-Lib (System Dependency)
# -----------------------------------------------------------------------------
# Note: Jules pre-installs many tools. We use sudo for system libs.
if [ ! -f "/usr/lib/libta_lib.so" ] && [ ! -f "/usr/local/lib/libta_lib.so" ]; then
    echo "⬇️ Installing TA-Lib C Library..."
    cd /tmp
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -q
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib
    ./configure --prefix=/usr > /dev/null
    make > /dev/null
    sudo make install > /dev/null
    cd ..
    rm -rf ta-lib*
    
    # CRITICAL: Go back to repo root to find requirements.txt later
    echo "🔙 Returning to repository root: $REPO_ROOT"
    cd "$REPO_ROOT"
else
    echo "✅ TA-Lib system library found."
fi

# 1.1 Configure Python Version (Jules default is 3.12, we need 3.10)
# -----------------------------------------------------------------------------
if command -v pyenv >/dev/null; then
    echo "🐍 Setting Python to 3.10 via pyenv..."
    pyenv install 3.10.18 -s  # Install only if missing
    pyenv local 3.10.18
    pyenv rehash
else
    echo "⚠️ pyenv not found, using system python (hope it is compatible)"
fi

# 2. Configure Environment Variables
# -----------------------------------------------------------------------------
echo "📝 Generating .env..."
cat <<EOF > .env
APP_ENV=development
TRADING_MODE=paper
DERIV_API_APP_ID=1089
DERIV_API_URL=wss://ws.binaryws.com/websockets/v3
LOG_LEVEL=DEBUG
EOF

# 3. Install Python Dependencies
# -----------------------------------------------------------------------------
pip install --upgrade pip
pip install ta-lib
pip install -r requirements.txt

# 4. CUSTOM FORK INSTALLATION (Clean Install)
# -----------------------------------------------------------------------------
echo "🔧 Installing custom python-deriv-api from Fork..."

# Use pip to install directly from git. keeping the repo clean.
# This avoids leaving a 'python-deriv-api' folder in your workspace.
pip install git+https://github.com/planetazul3/python-deriv-api.git@master

echo "✅ Fork installed successfully."

# 5. Verification
# -----------------------------------------------------------------------------
python -c "import deriv_api; print(f'✅ Success! Deriv API loaded from: {deriv_api.__file__}')"

# 6. CLEANUP (Crucial for Jules Verification)
# -----------------------------------------------------------------------------
# Jules requires a clean working tree to finish snapshot creation.
echo "🧹 Cleaning up workspace to satisfy Jules verification..."

# Remove local python version file created by pyenv
if [ -f ".python-version" ]; then
    rm ".python-version"
fi

# Final reset to ensure no untracked files or modifications remain
# This won't affect the installed packages in the environment
git reset --hard HEAD
echo "✨ Workspace is clean. Setup complete."
