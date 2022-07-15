#!/bin/bash
python -m src.utils.decoder.train \
            --test_results_folder ./results/decoder/example_ebbinghaus/ \
            --model_output_path ./models/decoder/example_ebbinghaus.pt \
            --train_dataset ./data/ebbinghaus/train_random_data \
            --test_datasets ./data/ebbinghaus/test_random_data ./data/ebbinghaus/test_small_flankers_data ./data/ebbinghaus/test_big_flankers_data

