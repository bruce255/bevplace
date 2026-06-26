#!/usr/bin/env bash
# Package project into a zip for uploading to Kaggle
set -e
ROOT_DIR="$(dirname "$0")"
cd "$ROOT_DIR"
ZIP_NAME="bevplace_project_$(date +%Y%m%d_%H%M%S).zip"
zip -r "$ZIP_NAME" . -x "*.git*" "runs/*" "cache*" "__pycache__/*"
echo "Created $ZIP_NAME in $ROOT_DIR"
