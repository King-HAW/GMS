#!/bin/bash

# bus
python train.py --config ./configs/bus_train.yaml

# busi
python train.py --config ./configs/busi_train.yaml

# glas
python train.py --config ./configs/glas_train.yaml

# ham10000
python train.py --config ./configs/ham10000_train.yaml

# kvasir-instrument
python train.py --config ./configs/kvasir-instrument_train.yaml
