#!/bin/bash

set -e
set -o pipefail

./main.py

# The mpeg4 codec is needed for windows
codec="-vcodec mpeg4 -q:v 0"
ffmpeg -framerate 15 -pattern_type glob -i "./output/hvs_at_*.png" -c:v libx264 -r 30 "$codec" out.mp4