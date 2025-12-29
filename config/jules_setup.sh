#!/bin/bash
# =============================================================================
# JULES INITIAL SETUP SCRIPT (FORK STRATEGY)
# Copy content of this file to: 
# Codebases -> x.titan -> Configuration -> Initial Setup
# =============================================================================
set -e

echo "🔧 Configuring environment for x.titan (Fork Strategy)..."

# 1. Install TA-Lib (System Dependency)
# -----------------------------------------------------------------------------
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
else
    echo "✅ TA-Lib system library found."
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

# 4. CUSTOM FORK INSTALLATION
# -----------------------------------------------------------------------------
echo "🔧 Setting up custom python-deriv-api fork..."

# We ensure we don't use the empty folder from the main repo
if [ -d "python-deriv-api" ]; then
    rm -rf python-deriv-api
fi

# CLONE YOUR FORK
# NOTE: Ensure you have added 'planetazul3/python-deriv-api' to Jules' allowed repos!
echo "⬇️ Cloning fork from planetazul3..."
git clone https://github.com/planetazul3/python-deriv-api.git

# Install
if [ -d "python-deriv-api" ]; then
    pip install -e ./python-deriv-api
    echo "✅ Fork installed successfully."
else
    echo "❌ Error: Clone failed. Please check permissions in Jules settings."
    exit 1
fi

# 5. Verification
# -----------------------------------------------------------------------------
python -c "import deriv_api; print(f'✅ Success! Deriv API loaded from: {deriv_api.__file__}')"
