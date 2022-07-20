# Decoder Approach

We attach 6 linear decoders at several stages of a `ResNet152` (including at the very input, and at the penultimate layer). 

With this approach, you can either do **classification** or **regression**. For example, the _ebbinghaus_ illusion is a regression task (decoder need to learn the size of the central circles), but many other tasks will require classification amongst classes.

You can use **the same** script for both regression or classification, with the same API. The type of folder arrangment will determine whether this is a regression or classifcation task, and the script will run the correct optimizaion and print the corrent info.

In both cases, you use the script in this way:
```
python -m src.utils.decoder.train \
            --test_results_folder results/name_dataset \
            --model_output_path models/name_dataset/model.pt \
            --train_dataset data/name_dataset/train_data \
            --test_datasets data/name_dataset/test_data1 data/name_dataset/test_data2 data/name_dataset/test_data3 
```
**Notice how we always start the script as a module, and from the root folder of this project, `mind-set`. Alway do that, otherwise it won't work.**

The `decoder.train` script will automatically train all decoder on the `train__data`  dataset, and will test **each test dataset separately**, providing appropriate info about them. The output will be the trained network, and a `.csv` file for each test dataset.


### Dataset for Classification

* **Example dataset `data/miniMNIST`**
* **Used in script `src/utils/decoder/train`**
* **Example usage in `src/utils/decoder/examples/classification.sh`**

This follows the [PyTorch arrangment](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) for image datasets: 
```
/root/class0/xxx.png
/root/class0/xxy.png
...
/root/class9/xxx.png
```

Example Usage:

```
python -m src.utils.decoder.train \
            --test_results_folder ./results/examples//miniMNIST/ \
            --model_output_path ./models/examples/miniMNIST.pt \
            --train_dataset ./data/examples/miniMNIST/training \
            --test_datasets ./data/examples/miniMNIST/testing1 ./data/examples/miniMNIST/testing2
```

### Dataset for Regression
* **Example dataset `data/ebbinghaus`** (generated it with `python -m src.ebbinghaus.generate_datasets`)
* **Used in script `src/utils/decoder/train`**
* **Example usage in `src/utils/decoder/examples/regression.sh`**


In this case each root folder will not contain other folders for classes (there are no classes!) but just `pngs`, which filename correspond to the independent variable. For example:
```
/root/0.21412.png
/root/23.3231.png
/root/2.12.png
...
```

Example Usage (we use a small version of the ebbinghaus dataset)
```

python -m src.utils.decoder.train \
            --test_results_folder ./results/examples/ebbinghaus/ \
            --model_output_path ./models/examples/ebbinghaus.pt \
            --train_dataset ./data/examples/ebbinghaus/train_random_data_100 \
            --test_datasets ./data/examples/ebbinghaus/test_random_data_50 ./data/examples/ebbinghaus/test_small_flankers_data_50 ./data/examples/ebbinghaus/test_big_flankers_data_50
```



### Training Loop Info
#### Logs
For classification, we print the following metrics during the training loop: `ema_loss`, `ema_acc`, `ema_acc_0`, `ema_acc_1`, ..., `ema_acc_5`. That is, a part from providing the loss, we provide the overall accuracy across all decoders, and the accuracy for each individual decoder. `ema` stands for [Exponential Moving Average](https://en.wikipedia.org/wiki/Moving_average) (to smooth across batches).
The output saved in the `csv` file are similarly named, without the `ema`, indicating that the values are simply the overall accuracy of the whole training set.

Similarly for regression, we compute the following metrics: 
`ema_loss`, `ema_rmse`, `ema_rmse_0`, ... `ema_rmse_5`. In this case `rmse` stands for [Root Mean Square Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation). As before, this is computed for each individual decoder (`rmse_x`) and across all decoders.  

#### Interrupting Training
Interrupting training with `CTRL+C` should actually end the training gracefully: the test dataset are computed one final time, the `csv` files are written in the appropriate folder, and the PyTorch model is also saved. In any case, a PyTorch `checkpoint` model is saved every epoch.

#### Other options
Check other otions with `python -m src.utils.decoder.train -h`. Let's try to all use the same default options when possible. If you want to do something that you can't do with the optional arguments, contact me!

## Neptune.ai
[Neptune.ai](www.neptune.ai) is a tool for logging experiments on the web. You don't need to use Neptune, but my training loop codebase is well integrated with it: it will save sample images and log training/testing charts in the Neptune page.
If you have an account, set `--neptune_proj_name` to your project name (which must be created through the UI). Also follow the instruction to install the neptune-client and setup the API.  



