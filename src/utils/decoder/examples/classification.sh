#!/bin/bash
## You can run the decoder example for a classification dataset, or for a regression dataset.
# A classification dataset is a folder containing a set of folders (one for each class).
python -m src.utils.decoder.train \
            --test_results_folder ./results/decoder/examples/miniMNIST/ \
            --model_output_path ./models/decoder/examples/miniMNIST.pt \
            --train_dataset ./data/examples/miniMNIST/training \
            --test_datasets ./data/examples/miniMNIST/testing1 ./data/examples/miniMNIST/testing2

# Note: I generally observe that all decoders learn this small miniMNISt apart from the last decoder, which accuracy stays at ~20%. We will maybe observe the same thing with bigger dataset, and the reason could be that the last decoder only has 2048 weights (against ~100k to ~800k of the other decoders), the rest of the net is frozen, so it might be too difficult to adapt those few weights to a new task. If that's the case, we might just exclude the last decoder.