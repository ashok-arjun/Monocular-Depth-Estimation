<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />


# Fully Convolutional Dense Networks for High-Quality Monocular Depth Estimation


This project addresses the problem of estimating the depth of a scene from the 2D image. A deep fully convolutional network in an encoder-decoder fashion has been used with skip connections between the encoder and the decoder.


# Results


## Qualitative

| Input RGB Image | Ground truth depth map | Our results|
|:---------------:|:----------------------:|:----------:|
|![](docs/image.png)|![](docs/gt.png)|![](docs/pred.png)|
|![](docs/image2.png)|![](docs/gt2.png)|![](docs/pred2.png)|

## Quantitative

|<img src="https://render.githubusercontent.com/render/math?math=\delta_1 \uparrow"> | <img src="https://render.githubusercontent.com/render/math?math=\delta_2 \uparrow"> |<img src="https://render.githubusercontent.com/render/math?math=\delta_3 \uparrow">|<img src="https://render.githubusercontent.com/render/math?math=rel \downarrow">|<img src="https://render.githubusercontent.com/render/math?math=rms\downarrow">|<img src="https://render.githubusercontent.com/render/math?math=log_{10}\downarrow">
| :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------:| :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: 
|0.852 | 0.976 | 0.995 | 0.122 | 0.500 | 0.053
# Instructions
<details>
<summary>
  <b>Installation</b> 
</summary>

To install, execute

```
pip install -r requirements.txt
```
  
</details>
<details>
<summary>
  <b>Data</b>
</summary>
  
[NYU Depth v2 train](https://tinyurl.com/nyu-data-zip)  - (50K images) (4.1 GB)

**The train zip file contains the ```data``` folder for training, and must be extracted.** 

[NYU Depth v2 test](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip) - (654 images) (1 GB)
  
**You don't have to extract the test zip file**, as the code loads the entire test zip file into memory.
</details>

<details>
<summary>
  <b>Training</b>
</summary>
  
The script ```train.py``` contains the code for training the model. It can be invoked with the following arguments:
  
```
usage: train.py [-h] --data_dir DATA_DIR --batch_size BATCH_SIZE
                --checkpoint_dir CHECKPOINT_DIR --epochs EPOCHS
                [--checkpoint CHECKPOINT] [--lr LR]
                [--log_interval LOG_INTERVAL]

Training of depth estimation model

mandatory arguments:
  --data_dir DATA_DIR   Train directory path - should contain the 'data'
                        folder
  --batch_size BATCH_SIZE
                        Batch size to process the train data
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save checkpoints in
  --epochs EPOCHS       Number of epochs
  
optional arguments:
  --checkpoint CHECKPOINT
                        Model checkpoint path
  --lr LR               Learning rate
  --log_interval LOG_INTERVAL
                        Interval to print the avg. loss and metrics

```

</details>

<details>
<summary>
  <b>Evaluation/Inference</b>
</summary>
  
The script ```evaluate.py``` contains the code for evaluating the model/for predicting the depth given an image. It can be invoked with the following arguments:

```

usage: evaluate.py [-h] --model MODEL [--data DATA] [--img IMG]
                   [--batch_size BATCH_SIZE] [--output_dir OUTPUT_DIR]

Evaluation of depth estimation model on either test data/own images

arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model checkpoint path
  --data DATA           Test data zip path(If evaluation on test data)
  --img IMG             Image path(If evaluation on a single image)
  --batch_size BATCH_SIZE
                        Batch size to process the test data
  --output_dir OUTPUT_DIR
                        Directory to save output depth images

```

</details>
