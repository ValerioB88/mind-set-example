# Mind Set Example Repo
Example for datasets that will require cosine similarity.
The general idea is that we want to measure the relative similarity of internal representation across different levels. For example, you want to check whether the internal representation of a "base" shape  is more similar to a shape with a metric property (MP) change vs a shape with a non-accidental-property (NAP) change. You might want to test that with many shapes and after several transformations. In this example the NAP vs MP are the levels. Different shapes are different samples
The way to organize the dataset is as follow: each file should be named `sampleName_level.png`. For example `triangle_NAP.png`, `triangle_MP.png`, `triangle_base.png`, `square_NAP.png`, etc. All samples should be in the same folder.


The main script you can take as an example is `run_cossim.py`. I use the `Config` class as a C-like struct for all experiment parameters. This will then be passed to `compute_cossim_from_img`. Here are some additional information for some of the parameters.

### Network_name
You can use many `network_name`s (check `net_utils.py->GrabNet()` for a list). Each network has an associated image size (224 for all networks and 299 for InceptionNet), which will be used for resizing the images, and a normalizaion valus, used for normalizing the samples. Since we are using networks pretrained on ImageNet, the ImageNet mean and standard deviation are used. 

### Pretraining
Use either "ImageNet" or "vanilla".

### Image_folder
The folder containing the samples.

### Affine_transf_code
You may want to augment the samples by using an affine transformation augmentation, and you want to make sure that the same transformation is applied on both pairs of the samples to which the cosine similarity is gonna be calculated. `t`, `s` and `r` indicate translation, scale and rotation. If you do not indicate anything default values will be used. For example you can pass `ts` to apply translation and scaling but no rotation. You can be more specific with the parameters, for example `t[-0.2,0.2]r[0,45]` will translate and rotate by the specified amount (translation parameters are fraction of total image size, rotation in degree)

## Background
Related to the affine transformation, if you specify a background here, it will "fill in" for the area outside the transformed image. Otherwise it will be white.

## Save_Layers
Indicate what layer type you want to compute the cosine similarity on.

## Rep
Each sample will be computed `rep` times, each time with a different affine transoatmion.

## Base_Name
When collecting the samples from the `image_folder` folder, the function `compute_cossim_from_img` will then compare the samples with the level equal to `base_name` with all the other levels. For example, in the NAP/MP example above, if the `base_name="base"` the function will compare `square_base.png` with `square_NAP.png`, and then `square_base.png` with `square_MP.png` and so on.

-----
`compute_cossim_from_img` will return a pandas dataframe, and a list of all layers which activation has been saved in the network.
This function will save a pickle object in the `config.result_folder`, and also a `debug_img` folder that shows some sample to which the cosine similarity has been computed. Really useful to check that it all makes sense.

`run_cossim.py` also does some simple analysis that should mostly work as an example to show you how to get some data out of the pandas dataframe.
Enjoy!