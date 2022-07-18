# MindSet
This repo is intended to be used by all members of the team involved in the MindSet project. Scripts for running a decoder task (either as a regression or classification method) and a cosine similarity task are provided. More might come if needed.

**_For more information about each approach, go to the README of the relevant folder: [cosine similarity approach](https://github.com/ValerioB88/mind-set-example/tree/master/src/utils/cosine_similarity) or [decoder approach](https://github.com/ValerioB88/mind-set-example/tree/master/src/utils/decoder)._**

If you just wanna mess around and try stuff use these scripts as you like (clone/fork/download/copy and paste). At some point, we need to understand more precisely how we are going to merge all our scripts (the ones we use for analysing our dataset). For now, you should take a look at the Contribution Guidelines below and follow them to make our life easier later on. Thanks!




**IMPORTANT: In all the examples, the working directory is ALWAYS the project root (`mind-set`), meaning that you need to run the scripts from the folder `mind-set`. Plus, to account for other modules dependencies, run it as a module. E.g. `python -m src.cosine_similarity_method.run_cossim_img_vs_folder`.**

## Contribution Guidelines
Put your dataset in `data/name_dataset`. If you have several variations of your dataset do something like `data/name_dataset/variation1`, `data/name_dataset/variation2`, etc.  If you use git, DO NOT ADD THE DATASET TO GIT. The dataset stays local on your machine. **We will decide later together how to share the datasets properly**. (The few datasets added currently are all currently for examples). 

The scripts for generating the dataset, for the analysis, and for any other thing that might be useful, go in `src/name_dataset/` (within this folder, arrange files and subfolders to your liking). In `src/name_dataset`, you should also put a bash file containing the command line used for that specific dataset, with a name indicating the approach it is referring to. See as an example `src/ebbinghaus/decoder_train.sh` containing the command line for training the decoder, or `src/NAPvsMP/cosine_similarity.sh` contains the loop for computing the cosine similarity and run the analysis on all the four `NAPvsMP` datasets in `data/NAPvsMP`.
When saving files, always save them in `results/name_dataset/`; when saving a model, save it in `models/name_dataset/`.

Overall, the general rule is to aim to mimick a similar folder structure across `data`, `src`, `models`, and `results`. 

## How should I create my dataset?
The instruction about the dataset structure to feed into each script are inside the README of the individual methods (cosine sim/decoder). However, if you are wondering whether there are any guidelines on the _images_ your dataset is made of, the answer is not really. I automatically apply a resize to 224x224 to each image, so generate something that looks good at that resolution. It's preferable to use a white stroke on a black background. A part from that, feel free to generate the image with your preferred method. Remember that you will need to share the script to generate the dataset.
