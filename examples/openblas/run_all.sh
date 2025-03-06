#!/usr/bin/bash
set -e
set -o pipefail

if [ -z "$1" ]; then
    echo "Usage: $0 <run_label>"
    exit 1
fi

RUN_LABEL=$1
OUTPUT_DIR="runs/$RUN_LABEL"
RESULT_DIR="results/$RUN_LABEL"

{

    mkdir -p $OUTPUT_DIR
#    source ./venv/bin/activate

    ./templatize.py ./config_mlkaps_template.json ./config_mlkaps.json

    echo "Running MLKAPS"
    mlkaps ./config_mlkaps.json -o $OUTPUT_DIR

    echo "Running exploration.py"
    ./exploration.py $RESULT_DIR

    # Perform extra-analysis here
} 2>&1 | tee run_all.log
