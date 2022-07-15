#!/bin/bash

python -m src.utils.cosine_similarity.compute_cossim_base_vs_folder \
            --base_image ./data/examples/closure/square.png \
            --folder ./data/examples/closure/angles_rnd \
            --result_folder ./results/examples/closure/full_vs_segm \
            --affine_transf_code "t[-0.2, 0.2]sr" \
            --repetitions 2  # For real experiment put at least 50

## After having run this, you would run your own analysis (there is no example of that here).
# To do that, start by opening the generated pickle file, which contains a pandas dataset and some metadata.