
## Cosine Similarity
All the relevant scripts are in `src/utils/cosine_similarity` and `src/utils`. Examples are in `src/utils/cosime_similarity/examples`.  

There are two ways to compute cosine similarity. One is by comparing _one base image_ with a set of images contained in a folder. The entry script is in `run_cossim_img_vs_folder`. The other one is to compare a set of images in a base folder with corresponding images in other folders. Entry script is `run_cossim_folder_vs_folder`. 


### Base image vs set of images
**Example in `/run_cossim_img_vs_folder_example.py`**

This is the simplest way of computing cosine similarity. Simply specify the path of the base image and the folder path. It will compute each pairwise similarities between the base image and each image in the folder. There are many optional argumentas that you can change in `ConfigSimple` (see [Config Options](#config-options)).  

### Folder vs Folder
**Example in`/run_cossim_folder_vs_folder.py`**

Here you compute cosine similarity between images in a base folder and corresponding images in other folders. For example, look at the `./data/NAPvsMP_standard/NAPvsMP` folder. Here, we want to compare each image in the `base` folder with each images in the other two folder. The image name have to match across folders: `NAPvsMP/base/1.png` will be compared with `NAPvsMP/NAP/1.png` and with `NAPvsMP/MP/1.png`. The image name doesn't need to be a number - but it needs to match across folders. You always need to indicate the name of the `base_folder` against which all other folders will be compared. A part from the example file, this is actually used in `src/NAPvsMP` folder


# Config Options
The`Config` class is something like a C-like struct for all experiment parameters. You can specify the `network_name`, (eg `vgg11bn`), the `pretraining` type (eg `ImageNet`) or path, the required `affine_transf(ormation)_code`, etc. 

### Network Name 
You can use many `network_name`s (check `net_utils.py->GrabNet()` for a list). Each network has an associated image size (224 for all networks and 299 for InceptionNet), which will be used for resizing the images, and  normalizaion values, used for normalizing the samples. Since we are using networks pretrained on ImageNet, the ImageNet mean and standard deviation are used as normalization values (again, in `net_utils.py->GrabNet()`).

### Pretraining
Use either "ImageNet" or "vanilla" or a path for a pytorch state_dict model file. 

### Image_folder
The folder containing the samples.

### Affine_transf_code
You may want to augment the samples by using an affine transformation augmentation. 
 `t`, `s` and `r` indicate that you want to apply translation, scale and rotation. If you do not indicate anything, default values will be used. For example you can pass `ts` to apply translation and scaling but no rotation. You can be more specific with the parameters, for example `t[-0.2,0.2]r[0,45]` will translate and rotate by the specified amount (translation parameters are fraction of total image size, rotation in degree). 
Since the comparison here is done in pair, you might often want these transformations to match across the two pairs. To do that, set the `matching_transform` argument to True.

### Background
Related to the affine transformation, if you specify a background here, it will "fill in" for the area outside the transformed image. Otherwise it will be white.

### Save_Layers
Indicate what layer type you want to compute the cosine similarity on. In practice, we check whether the layer name contains any of the string indicated here. For example, you can use "MaxP", "ReLU", etc., to save these layers.

### Rep
Each sample will be computed `rep` times, each time with a different affine transformation.


## Output
`compute_cossim_from_img` will return a pandas dataframe, and a list of all layers which activation has been used for computing the cosine similarity.
This function will save the dataframe as a pickle object in the `config.result_folder`, and also a `debug_img` folder that shows some of the pairs used for computing cosine similarity. Really useful to check that it all makes sense, expecially with respect to the affine transformations.

The example file `run_cossim_folder_vs_folder.py` also does some simple analysis that should mostly work as an example to show you how to get some data out of the pandas dataframe.

