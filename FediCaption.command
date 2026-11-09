#!/usr/bin/env bash
# Double-clickable macOS Finder launcher for FediCaption
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
exec python3 "$DIR/fedicaption" "$@"
