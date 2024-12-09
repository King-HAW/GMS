#!/bin/bash

# bus
python valid.py --config ./configs/bus_valid.yaml

# busi
python valid.py --config ./configs/busi_valid.yaml

# glas
python valid.py --config ./configs/glas_valid.yaml

# ham10000
python valid.py --config ./configs/ham10000_valid.yaml

# kvasir-instrument
python valid.py --config ./configs/kvasir-instrument_valid.yaml
