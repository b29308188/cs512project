#!/bin/bash

rm -rf results 2>/dev/null
mkdir results

python3 bin/processing.py rawData results
