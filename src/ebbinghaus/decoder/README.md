# Ebbinghaus Decoder Approach
In this approach, we add a linear layer at several stages on a network (I used a ResNet152, see `ResNet152decoders`), then train just these decoders on a task, and test it on a similar task.

In this folder, we apply this approach to the Ebbinghaus illusion. There will be enough differences across different psychological tasks that it doesn't really make sense to make the `one script to rule them all`. Instead, you can use my `train` and `test` file as a template for your version, and change each bit as needed.

The most important bit, the one that you will probably want to change, is the Dataset loader. In `train.py`, the `EbbinghausTrain` class generates Ebbinghaus samples one the fly. These are red circles of different sizes, with some random white circles around it. Each decoder needs to learn to output the size (in fraction of the full canvas) of the red circles.

In `test.py`, we test the network on the well known ebbinghaus illusion, one with big white circles (flankers) and one with small flankers, and check whether they estimate on average samples with big flankers to be smaller than those with small flankers. 


## Coding Quirks
I use mostly well known python functions and libraries, but there are two quirks that might confuse readers:
- my code is well integrated with the experiment tracking service [Neptune.ai](www.neptune.ai). If you have an account, set the `weblogger` option in `Config` equal to `True`, otherwise `False`. This will save sample images and log training/testing charts in the Neptune page
- In order to plot and log things properly, I add some paramters to the the dataset (see `src/ebbinghaus_illusion/decoder/test.py`, the `add_info_to_dataset`). When you use your own dataset, you always need to add these values: `name_ds` (the name of the dataset, can be anything you want), `stats` (the statistics used for normalizing the images - we generally use ImageNet stats) and `transform` (the typical pytorch transform). Not adding even just one of this will break things. 








