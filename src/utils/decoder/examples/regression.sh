#!/bin/bash

# You can run the decoder example for a classification dataset, or for a regression dataset.
# A regression folder is a folder containing only images. The images
python -m src.ebbinghaus.generate_datasets --num_training_data 600 --num_testing_data 200 --folder ./data/examples/ebbinghaus

python -m src.utils.decoder.train \
            --test_results_folder ./results/examples/ebbinghaus \
            --model_output_path ./models/examples/ebbinghaus.pt \
            --train_dataset ./data/examples/ebbinghaus/train_random_data_600 \
            --test_datasets ./data/examples/ebbinghaus/test_random_data_200 ./data/examples/ebbinghaus/test_small_flankers_data_2000 ./data/examples/ebbinghaus/test_big_flankers_data_200

