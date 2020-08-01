# Fully Convolutional Dense Networks for High-Quality Monocular Depth Estimation


 *Estimating depth from a single RGB image is an ill-posed problem that requires both global and local information. This problem plays an important role in various applications including autonomous driving and scene understanding. We tackle this problem by leveraging transfer learning and by using an encoder-decoder architecture that is trained end-to-end. A combination of three suitable losses has been used for optimization. We demonstrate through careful ablation studies that our network produces comparable results on the NYU Depth v2 dataset and captures the object boundaries faithfully.*


# Results


## Qualitative

(images comparison)

## Quantitative

(metrics table)

# Pre-trained models

[Pretrained Model](https://1drv.ms/u/s!AlYxAnPCKqHhgxNdQLu_icSxf-rs?e=lF4CMH) - 132 MB

# Instructions

<details>
<summary>
Data
</summary>
* NYU Depth v2 **train** (50K images) (4.1 GB)  
* NYU Depth v2 **test** (654 images) (1 GB)
  
You don't need to extract the zip file, as the code loads the entire zip file into memory when training
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
