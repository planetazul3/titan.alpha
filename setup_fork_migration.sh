#!/bin/bash
# =============================================================================
# MIGRATION HELPER: Local Folder -> Git Fork
# Run this script to prepare your local environment
# =============================================================================

FORK_URL="git@github.com:planetazul3/python-deriv-api.git"
BACKUP_DIR="python-deriv-api-legacy_backup"

echo "🚀 Starting Migration to Fork..."

# 1. Backup existing custom code
if [ -d "python-deriv-api" ]; then
    echo "📦 Backing up current folder to $BACKUP_DIR..."
    rm -rf $BACKUP_DIR
    mv python-deriv-api $BACKUP_DIR
else
    echo "❌ Error: 'python-deriv-api' folder not found!"
    exit 1
fi

# 2. Clone fresh official repo (to get clean git history)
echo "⬇️ Cloning official upstream repository..."
git clone https://github.com/deriv-com/python-deriv-api.git
cd python-deriv-api

# 3. Setup Remotes
echo "🔗 Configuring remotes..."
git remote rename origin upstream
git remote add origin $FORK_URL

# 4. Instructions
echo ""
echo "======================================================================="
echo "✅ PREPARATION COMPLETE!"
echo "Now you must perform the following manual steps to finish:"
echo ""
echo "1. Go to GitHub and create a NEW empty repository named 'python-deriv-api'"
echo "2. Copy your custom modified files FROM: ../$BACKUP_DIR"
echo "   TO:   ./python-deriv-api"
echo "   (Be careful not to overwrite .git folder!)"
echo "3. Run: git add . && git commit -m 'Port custom changes' && git push -u origin master"
echo "======================================================================="
