#!/bin/bash

# Calculate directory local to script
SCRIPTS_DIR="$( cd "$( dirname "$0" )" && pwd )"

cd $SCRIPTS_DIR/..
PYTHONPATH=$PWD python3 -m experiments.auto_arima.error_between_medoids $@
