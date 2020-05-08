#!/bin/bash

# Calculate directory local to script
SCRIPTS_DIR="$( cd "$( dirname "$0" )" && pwd )"

# get a clean source
$SCRIPTS_DIR/clean.sh

# Run tests using pytest
cd $SCRIPTS_DIR/..
PYTHONPATH=. pytest-3 --pyargs spta -v
