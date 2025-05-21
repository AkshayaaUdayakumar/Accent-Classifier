#!/bin/bash
echo "Starting Python dependency installation..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_PATH="${SCRIPT_DIR}/requirements.txt"

echo "Looking for requirements.txt at: ${REQUIREMENTS_PATH}"

# Check if requirements.txt exists
if [ ! -f "${REQUIREMENTS_PATH}" ]; then
    echo "WARNING: requirements.txt not found at ${REQUIREMENTS_PATH}"
    echo "Creating minimal requirements.txt"
    # Create a minimal requirements file with just the essentials
    cat > "${REQUIREMENTS_PATH}" << EOL
numpy
pytube
pydub
EOL
fi

# Try to install the minimal dependencies we need
if command -v pip &> /dev/null; then
    echo "Using pip"
    pip install --no-cache-dir numpy pytube pydub
elif command -v pip3 &> /dev/null; then
    echo "Using pip3"
    pip3 install --no-cache-dir numpy pytube pydub
elif command -v python -m pip &> /dev/null; then
    echo "Using python -m pip"
    python -m pip install --no-cache-dir numpy pytube pydub
elif command -v python3 -m pip &> /dev/null; then
    echo "Using python3 -m pip"
    python3 -m pip install --no-cache-dir numpy pytube pydub
else
    echo "WARNING: pip command not found. Continuing without installing Python dependencies."
fi

echo "Python dependency installation completed"

# For Vercel deployments, we'll create a special file that tells our API routes to use the simplified classifier
if [ -n "$VERCEL" ] || [ -n "$VERCEL_ENV" ]; then
    echo "Detected Vercel environment - using simplified accent classifier"
    
    # Make the simplified classifier executable
    chmod +x "${SCRIPT_DIR}/accent_classifier_simple.py"
    
    # Create a symlink from accent_classifier.py to accent_classifier_simple.py
    # This ensures our API routes will use the simplified version without code changes
    ln -sf "${SCRIPT_DIR}/accent_classifier_simple.py" "${SCRIPT_DIR}/accent_classifier.py"
    
    # Create a marker file to indicate we're in Vercel
    touch "${SCRIPT_DIR}/.vercel_deployment"
    
    echo "Set up simplified accent classifier for Vercel deployment"
fi

# Always create the mock/fallback version as ultimate backup
chmod +x "${SCRIPT_DIR}/accent_classifier_mock.py"
echo "Created fallback accent classifier in case of dependency issues"

