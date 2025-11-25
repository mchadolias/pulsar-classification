#!/bin/bash
# Simple Hugging Face Sync Script
# Edit the variables below for your project
#  HF_USERNAME="their-username"
#  HF_SPACE="their-space-name" 
#  HF_DIR="where/to/store/hf/files" 

# CONFIG - Edit these for your project
HF_USERNAME="mchadolias"
HF_SPACE="pulsar-classification-htru2"
HF_DIR="deployment/huggingface-space"

echo "🔄 Syncing to Hugging Face Space: $HF_USERNAME/$HF_SPACE"

# Setup
cd "$(dirname "$0")/.."  # Go to project root
export GIT_ASKPASS=""

# Clone or update repo
if [ -d "$HF_DIR" ]; then
    cd "$HF_DIR"
    git pull origin main
else
    git clone "git@hf.co:spaces/$HF_USERNAME/$HF_SPACE" "$HF_DIR"
    cd "$HF_DIR"
fi

# Copy files
cp ../deployment/predict.py ./app.py
cp ../deployment/huggingface/requirements.txt ./
cp ../outputs/models/pipeline.pkl ./models/

# Commit and push
git add .
git commit -S -m "Update: $(date +'%Y-%m-%d %H:%M')"
git push origin main

echo "✅ Done! Your Space: https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"