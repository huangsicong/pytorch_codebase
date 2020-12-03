## Evaluating Lossy Compression Rates of Deep Generative Models
The general purpose codebase based on the code used in the ICML paper: Evaluating Lossy Compression Rates of Deep Generative Models

**Original Authors**: [Sicong Huang](https://www.cs.toronto.edu/~huang/), [Alireza Makhzani*](http://www.alireza.ai/), [Yanshuai Cao
](http://www.cs.toronto.edu/~g8acai/index.html), [Roger Grosse](https://www.cs.toronto.edu/~rgrosse/) (*Equal contribution)

The Full codebase can be found here: [https://github.com/huangsicong/rate_distortion](https://github.com/huangsicong/rate_distortion)

Special thanks to Gerald Shen for improving this codabase with Sicong Huang. 

## Citing this work
```
@article{huang2020rd,
  title={Evaluating Lossy Compression Rates of Deep Generative Models},
  author={Huang, Sicong and Makhzani, Alireza and Cao, Yanshuai and Grosse, Roger},
  booktitle = {ICML},
  year={2020}
}
```

## Running this code
Dependencies are listed in requirement.txt. 
Lite tracer can be found [here](https://github.com/BorealisAI/lite_tracer).

There are only 3 argparse arguments: 
- `hparam_set`: (str) This is label of the experiment, and it point to a set of hyper parameters associated with this experiment, organzied by an Hparam object. They are registered under [codebase/hparams](codebase/hparams).
- `e_name`: (str) "Extra name". Used to add an extra suffix after the hparam_set in the name of the experiment. This is used to run another copy (for testing purposes for example) of the experiment defined by hparam_set without having to create the same hparam_set. 
- `overwrite` (boolean): if set to true it will overwrite the previous result directory of the same experiment if it exists

The configuration for each experiment is defined by an Hparam object registered in  [codebase/hparams](codebase/hparams). The default value for an undefined field is **None**. The Hparam object is hierarchical and compositional for modularity. 


This codebase has a self-contained system for keeping track of checkpoints and outputs based on the Hparam object. To load checkpoint from another experiment registered in the codebase, assign **load_checkpoint_name** to the name of a registered **hparam_set** in the codebase. If the model you want to test is not trained with this codebase, to load your model, you can simply set **specific_model_path** to the path of your decoder weights. 

To run the codebase in Google Colab see: [Notebook](https://colab.research.google.com/drive/1X7_FM0pRSQt7TJLJgkDJXdAiltvetzcN?usp=sharing). This by default will clone the master branch and run the codebase using a test command. Consider using this file to get started with running your custom experiments on Colab.

Example run to train a DCGAN based VAE locally or on a compute cluster:
  ```
  python -m codebase.train_gen --hparam_set=dcvae100_mnist
  ``` 
[codebase/train_gen.py](codebase/train_gen.py) is a root script that is written to process a type of task, in this case, the training of generative models. 

## Reproducing our results.
Note that in this codebase the AIS and rate distortion code was taken out, so in this version of the codebase, it can only go as far as getting the trained generative models ready to use. 

- Set the paths. 
  
  Make sure to properly set all the directories and paths accordingly, including the output directories **output_root_dir**, the data directory **data_dir** and the checkpoint directory **checkpoint_root_dir** in the [codebase/hparams/defaults.py](codebase/hparams/defaults.py). Note that the datasets will be automatically downloaded if you don't have then in the **data_dir** the already. For all the sbatch files, make sure to set the working directory accordingly (to where this repo is cloned) in each command as well. 

- Get the checkpoints. 
  
  The training code and script for VAEs are included, for the rest of the models, trained checkpoints can be found [here](https://drive.google.com/drive/folders/19tqmlGm5oMGWtlAPcLcZdisGo30xrR4p?usp=sharing).
  Set the **FILEPATH** (it should end with .zip) and run this command to download the checkpoints zip: 
  ```
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KyIGHCIDl4DDRBLBcaBg39adsMn4xAev' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KyIGHCIDl4DDRBLBcaBg39adsMn4xAev" -O FILEPATH && rm -rf /tmp/cookies.txt
  ```
  And then unzip it into the **checkpoint_root_dir**
  ```
  unzip FILEPATH -d checkpoint_root_dir
  ```

  If you have access to a compute cluster that has slurm and sbatch installed, you can run the sbatch files to reproduce our experimental results. All **.sh** files are sbatch files in [codebase/hparams](codebase/hparams), to run them:
  ```
  sbatch FILENAME.sh
  ```
  Notice that if you don't already have the datasets downloaded, running multiple runs concurrently might cause interference of the data loading/downloading processes. So a good practice is to let the data downloading process finish before starting another runs related to the same dataset. 
- Ploting script
  You will need to modify [codebase/plot.py](codebase/plot.py) for your own purposes. The idea here is that the script could automatically extract data based on the experiment name. Set the **output_root_dir** in your plotting hparam to be the same as the **output_root_dir** in [codebase/hparams/defaults.py](codebase/hparams/defaults.py) which was set in the beginning to custimize the plots.
  Then run **icml_plots.sh**.



## Test your own generative models. 
The codebase is also modularized for testing your own decoder-based generative models. You need to register your model under [codebase/models/user_models.py](codebase/models/user_models.py), and register the Hparam object at [codebase/hparams/user_hparams.py](codebase/hparams/user_hparams.py). Your model should come with its decoder variance model.x_logvar as a scalar or vector tensor. Set **specific_model_path** to the path of your decoder weights.

### PyTorch: 
If the generative models is trained in PyTorch, the checkpoint should contain the key "state_dict" as the weights of the model.

### Others:
If the generative models is trained in other framewords, you'll need to manuually bridge and load the weights. For example, the AAEs were trained in tensorflow, with the weights saved as numpy, and then loaded as nn.Parameter in PyTorch. Refer to [codebase/utils/aae_utils.py](codebase/utils/aae_utils.py) for more details.

## Detailed Experimental Settings
More details on how to control experimental settings can be found below. 


General configuration:

- `specific_model_path`: (str) Set to the path to the decoder weights for your own experiments. Set to None if you are reproducing our experiments.
- `original_experiment`: (boolean) This should be set to **True** when the checkpoint or the model is from the paper. When you are testing your own generative model, set this False and it will load from `specific_model_path` instead of the directories generated by this codebase. You may need to customize the **load_user_model** function in [codebase/utils/experiment_utils.py](codebase/utils/experiment_utils.py) for your own generative model.   
- `output_root_dir`: (str) The root dir for the experiment workspace. Experiment results, checkpoints will be saved in subfolders under this directory.
- `group_list`: (list) A list specifying the file tree structure for the output of this experiment, inside the `output_root_dir`.   
- `step_sizes_target`: (str) When not defined or set to None, HMC step sizes adaptively tunned and saved during AIS. 
  When specified as the name of hparam_set of another previously finished experiment, 
  HMC step size will be loaded from that experiment.

- `model_name`: (str) The name of the model you want to use. The model must be registered under [codebase/models](codebase/models).

Sub-hparams: 

- `model_train` contains information about the original training setting of the model. (In this code base only VAE training is supported) 
- `rd`: contains information about the AIS setting for the rate distortion curve.
- `dataset`: contains information about the dataset. Set mnist() for MNIST and cifar() for CIFAR10. 

**model_train** sub-Hparam:
  - `z_size`: (int) Size of the latent code.
  - `batch_size`: (int) The batch size for training. 
  - `epochs`: (int) The number of epochs to train. 
  - `x_var`: (float) Initial decoder variance.

**dataset** sub-Hparam
  - `name`: (str) The name of the dataset. 
  - `train`: (str) Load the train set if True, test set otherwise. 
  - `input_dims`: (list) A list specifying the input dimensions. 
  - `input_vector_length`: (int) The product of **input_dims**.
  

The rest: normally the below settings do not need to be changed.
  - `cuda`: (boolean) Whether or not to use CUDA. 
  - `verbose`: (boolean) Verbose or not for logging and print statements. 
  - `random_seed`: (int) Random seed.  
  - `n_val_batch`: (int) Number of batch you want to test on during training or IWAE. During training it'll test on a held-out validation set. 
  - `no_workers`: (int) number of processes to load our data 