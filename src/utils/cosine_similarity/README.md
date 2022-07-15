
## Cosine Similarity

[//]: # (All the relevant scripts are in `src/utils/cosine_similarity` and `src/utils`. Examples are in `src/utils/cosime_similarity/examples`.  )

There are two ways to compute cosine similarity. One is by comparing _one base image_ with a set of images contained in a folder (`compute_cossim_base_vs_folder.py`). The other one is to compare a set of images in a base folder with corresponding images in other folders (`compute_cossim_across_folders.py`). You will find an example for each one in the folder `src/utils/cosine_similarity/examples/`. 
Next is how to organize the dataset to use the scripts:


### Dataset: Base image vs set of images

* **Example dataset in `data/examples/closure`**
* **Used in script `src/utils/cosine_similarity/compute_cossim_base_vs_folder.py`** 
* **Example usage in `src/utils/cosine_similarity/examples/cossim_base_vs_folder.sh`**

This is the simplest way of computing cosine similarity. Simply specify the path of the base image and the folder containing images you want to compare the base image with. The `compute_cossime_base_vs_folder.py` will take this type of dataset. An example of this dataset is `data/examples/closure/`, which contains the base image `square.png` and a folder with a bunch of other images (each image names here don't matter). 

**Example usage:**

```
python -m src.utils.cosine_similarity.compute_cossim_base_vs_folder \
            --base_image ./data/examples/closure/square.png \
            --folder ./data/examples/closure/angles_rnd \
            --result_folder ./results/examples/closure/full_vs_segm \
            --affine_transf_code "t[-0.2, 0.2]sr" \
            --repetitions 50
```

The meaning of the `--affine_transf_code` optional args is explained below. 

**(Notice how we always start the script as a module, and from the root folder of this project, `mind-set`. Alway do that, otherwise it won't work)**
### Dataset: Folder vs Folder
* **Example dataset in `data/NAPvsMP/NAPvsMP_standard`**

* **Used in the script `src/utils/cosine_similarity/compute_cossim_acros_folders.py`**

* **Example usage `src/utils/cosine_similarity/examples/cossim_across_folders.sh`**


Here you compute cosine similarity between images in a base folder and corresponding images in other folders. An example is `/data/NAPvsMP/NAPvsMP_standard` folder. Here, we want to compare each image in the `base` folder with each images in the other two folder. The image name have to match across folders: `NAPvsMP_standard/base/1.png` will be compared with `NAPvsMP_standard/MP/1.png` and with `NAPvsMP_standard/NAP/1.png`. The image name doesn't need to be a number - but it needs to match across folders. You always need to indicate the name of the `base_folder` against which all other folders will be compared. The script `compute_cossim_across_folder` will compare the base folder with any number of folders.

**Example Command Line usage**:
```
python -m src.utils.cosine_similarity.compute_cossim_across_folders \
                            --base_folder_name NS \
                            --folder ./data/NAPvsMP/NAPvsMPlines/ \
                            --result_folder ./results/NAPvsMP/NAPvsMPlines/ \
                            --affine_transf_code "t[-0.2, 0.2]sr" \
                            --repetitions 2  # For real experiment put at least 50
```

### Optional Arguments
We should aim to keep experiments consistent, so I advice to generally stick with the default arguemnts. However, for your own entertainment you can explore different setups. A full list of optional arguments can be seen by running `python -m src.utils.cosine_similarity.compute_cossim_across_folders -h`. Here are some of them that deserve a longer explanation.

### Affine_transf_code
You may want to augment the samples by using an affine transformation augmentation. Use the `--affine_transf_code` for that.
 `t`, `s` and `r` indicate that you want to apply translation, scale and rotation. If you do not indicate anything, default values will be used. For example you can pass `ts` to apply translation and scaling but no rotation. You can be more specific with the parameters, for example `t[-0.2,0.2]r[0,45]` will translate and rotate by the specified amount (translation parameters are fraction of total image size, rotation in degree). 
By default, the *same* transformation is applied across all pairs of a comparison. If that's not what you want, set the optional argument `--matching_transform` to `False`.

Another related optional argument is `--affine_transf_background`. The colour specified here (default is `black`) will "fill in" the area outside the transformed image (for example, when you apply a rotation to the image, s. Otherwise it will be white.

### Save_Layers
`--save_layers` Indicate what layer type you want to compute the cosine similarity on. In practice, we check whether the layer name contains any of the string indicated here. For example, you can specify `--set_layers MaxP ReLU Conv2d`. Default is `Conv2d Linear`.


## Output
Both `compute_cossim_across_folders` and `compute_cossim_base_vs_folder` will return a pandas dataframe, and a list of all layers which activation has been used for computing the cosine similarity. 
These scripts will save the dataframe as a pickle object in the `--result_folder`. They'll also write images in a `debug_img` folder (within the `--result_folder`)  showing samples of the pairs used for computing cosine similarity. Really useful to check that it all makes sense, expecially with respect to the affine transformations.

## Sample Analysis
There is a sample analysis script for the comparison across folder: `analysis_across_folder.py`. It does some simple analysis that should mostly work as an example to show you how to get some data out of the pandas dataframe. The same analysis file is present in a `.ipynb` format in case you prefer that 
The example file `src/utils/cosine_similarity/cossim_across_folders.sh` show how to pipe with the script computing the cosine similarity. You are free to setup the analysis as you please. As usual, keep all contained within the same folder in `src`, and stick to the same folder structure for `src`, `data`, and   `results`.

