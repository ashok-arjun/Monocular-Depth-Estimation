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
|0.842|0.969|0.992|0.125|0.499|0.053


# Pre-trained models

[Pretrained Model](https://1drv.ms/u/s!AlYxAnPCKqHhgxNdQLu_icSxf-rs?e=lF4CMH) - 132 MB

# Instructions

<details>
<summary>
Data
</summary>
  
[NYU Depth v2 train](https://tinyurl.com/nyu-data-zip)  - (50K images) (4.1 GB)
    
[NYU Depth v2 test](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip) - (654 images) (1 GB)
  
You don't have to extract the zip files, as the code loads the entire zip file into memory when training
</details>

<details>
<summary>
Training and inference
</summary>
The following code can be executed in the root directory to train and simultaneously validate(in a notebook or in a python script file). The config values can be changed. 

```
from train import Trainer

config = {}
config['batch_size'] = 8 
config['lr] = 3e-4
config['test_batch_size'] = 2
config['epochs'] = 15            
config['lr_scheduler_step_size'] = 5

trainer = Trainer('path_to_train_zipfile.zip', resized = True)
trainer.train_and_evaluate(config, resume_checkpoint_file_name, local)
```

The following code can be executed to evaluate on the test dataset:

```
from model.net import DenseDepth
from model.dataloader import get_test_data
from evaluate import evaluate_list
from utils import *

model = DenseDepth()
load_checkpoint('pretrained_model_path.pth.tar', model)
samples, crop = get_test_data('path_to_test_zip_file.zip') 
test_metrics = evaluate_list(model, samples, crop, test_batch_size, resized) # evaluate list can be modified to return the predictions, if required
```

</details>
