#!/bin/bash

python3 ./baselineFIQA/main.py  -c ./baselineFIQA/baseline_FIQA_config.yaml

python3 ./observer/initialization.py -c ./observer/initialization_config.yaml

python3 ./observer/inference.py -c ./observer/inference_config.yaml

python3 delete_me.py