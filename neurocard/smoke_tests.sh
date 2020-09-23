#!/bin/bash
set -ex

python made.py
python transformer.py
python null_semantics_test.py
python run.py
