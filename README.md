# Mind Set Example Repo
This Repo can be used for those tasks requiring the cosine similarity or the decoder approach. This should cover the majority of the examples in the mind set repo.


**_For more information about each approach, go to the README of the relevant folder: [cosine similarity approach](https://github.com/ValerioB88/mind-set-example/tree/master/src/utils/cosine_similarity) or [decoder approach](https://github.com/ValerioB88/mind-set-example/tree/master/src/ebbinghaus/decoder)._**

## General Usage
**IMPORTANT: In all the examples, the working directory is ALWAYS `/MindSetExample`, meaning that you need to run the scripts from `/MindSetExample`. To account for the dependencies, run it as a module. E.g. `python3 -m src.cosine_similarity_method.run_cossim_img_vs_folder`.**



## Coding Quirks
I use mostly well known python functions and libraries, but there are two quirks that might confuse readers:
- my code is well integrated with the experiment tracking service [Neptune.ai](www.neptune.ai). If you have an account, set the `weblogger` option in `Config` equal to `True`, otherwise `False`. This will save sample images and log training/testing charts in the Neptune page
- I do not use command line arguments. Instead, most scripts contain a `Config` object (or `ConfigSimple`) that contains all the needed parameters for an experiment. Each experiment corresponds to a different Config object (instead of a different command line). For your own simulation you will use a different python file with its own `Config` object. You will find some examples for both the cosine (`cosine_similarity/run_cossim_img_vs_folder`) and the decoder method (`ebbinghaus_illusion/decoder/train.py`).


## To Do
Ideally the datasets in `./data` should only contain dataset needed to run the experiments. Right now it contains a lot of crap - I will fix that soon! 