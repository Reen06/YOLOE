#!/bin/bash
set -e

# Change to app directory
cd /app

# Initialize Git LFS if .gitattributes exists
if [ -f .gitattributes ]; then
    echo "Initializing Git LFS..."
    git lfs install --force
    
    # Check if model files are pointers and pull if needed
    if [ -f yoloe-11s-seg.pt ]; then
        if head -n 1 yoloe-11s-seg.pt | grep -q "version https://git-lfs"; then
            echo "Model files are Git LFS pointers. Pulling actual files..."
            git lfs pull || echo "Warning: Git LFS pull failed. Model files may not be available."
        fi
    fi
fi

# Execute the command passed to the container
exec "$@"

