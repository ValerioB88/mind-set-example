#!/bin/bash

python -m src.utils.cosine_similarity.compute_cossim_across_folders \
                            --base_folder_name NS \
                            --folder ./data/NAPvsMP/NAPvsMPlines/ \
                            --result_folder ./results/NAPvsMP/NAPvsMPlines/ \
                            --affine_transf_code "t[-0.2, 0.2]sr" \
                            --repetitions 2  # For real experiment put at least 50

## After having run this, you can run the related notebook, or you can ran it as a script (they do the same things) in the following way:
python -m src.utils.cosine_similarity.examples.analysis_across_folders \
                            --pickle_path ./results/NAPvsMP/NAPvsMPlines/cossim_df.pickle \
                            --result_folder ./results/NAPvsMP/NAPvsMPlines/


## Feel free to write your own analysis.
