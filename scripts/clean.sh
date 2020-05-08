#!/bin/bash

# Calculate directory local to script
BIN_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $BIN_DIR/..

# remove all these transient files
find -name '*.pyc' -delete
find -name __pycache__ -delete
rm -rf .pytest_cache
rm -rf build
