## Evaluating Lossy Compression Rates of Deep Generative Models
The codebase template used in the ICML paper: Evaluating Lossy Compression Rates of Deep Generative Models

**Author**: [Sicong Huang](https://www.cs.toronto.edu/~huang/), [Alireza Makhzani*](http://www.alireza.ai/), [Yanshuai Cao
](http://www.cs.toronto.edu/~g8acai/index.html), [Roger Grosse](https://www.cs.toronto.edu/~rgrosse/) (*Equal contribution)

The Full codebase can be found here: The Full codebase can be found here: [https://github.com/huangsicong/rate_distortion](https://github.com/huangsicong/rate_distortion)

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

There are only 2 argparse arguments: 
- `hparam_set`: (str) This is label of the experiment, and it point to a set of hyper parameters associated with this experiment, organzied by an Hparam object. They are registered under [codebase/hparams](codebase/hparams).
- `e_name`: (str) "Extra name". Used to add an extra suffix after the hparam_set in the name of the experiment. This is used to run another copy (for testing purposes for example) of the experiment defined by hparam_set without having to create the same hparam_set. 


The configuration for each experiment is defined by an Hparam object registered in  [codebase/hparams](codebase/hparams). The default value for an undefined field is **None**. The Hparam object is hierarchical and compositional for modularity. 



This codebase has a self-contained system for keeping track of checkpoints and outputs based on the Hparam object. To load checkpoint from another experiment registered in the codebase, assign **load_hparam_name** to the name of a registered **hparam_set** in the codebase. If the model you want to test is not trained with this codebase, to load your model, you can simply set **specific_model_path** to the path of your decoder weights. 

To run the codebase in Google Colab see: [Notebook](https://colab.research.google.com/drive/1X7_FM0pRSQt7TJLJgkDJXdAiltvetzcN?usp=sharing). This by default will clone the master branch and run the codebase using a test command. Consider using this file to get started with running your custom experiments on Colab.


## Reproducing our results.

- Set the paths. 
  
  Make sure to properly set all the directories and paths accordingly, including the output directories **output_root_dir**, the data directory **data_dir** and the checkpoint directory **checkpoint_root_dir** in the [codebase/hparams/defaults.py](codebase/hparams/defaults.py). Note that the datasets (MNIST and CIFAR10) will be automatically downloaded if you don't have then in the **data_dir** the already. For all the sbatch files, make sure to set the working directory accordingly (to where this repo is cloned) in each command as well. 

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

- Two rounds of AIS runs. 

  To conform to mathematical correctness when tuning the HMC step sizes, for each setting there are two round of experiments. During the first round, step sizes of HMC are adaptively tuned and saved. During the second round, step sizes are loaded and the rate distortion curve (or BDMC gap) was computed with another random seed.

  All experiments used in the paper can be found inside [codebase/hparams](codebase/hparams), and you can reproduce all experiments by running the sbatch files in a particular order. The ordering is important becasue there are dependencies between different runs. The first round of HMC step-size tuning experiments must be finished before the second round of runs can load step sizes.

- First set of experiments<br />
  All **.sh** files are sbatch files in [codebase/hparams](codebase/hparams) and to run them, simply
  ```
  sbatch FILENAME.sh
  ```
  Notice that if you don't already have the datasets downloaded, running multiple runs concurrently might cause interference of the data loading/downloading processes. So a good practive is to let the data downloding process finish before starting another runs related to the same dataset. This is taken care of by the following sbatch files and again it's important to wait for one to finish before starting the next. First run **analytical.sh** which download MNIST and train a linear VAE. You can also use this to check whether everything is set up properly. Then run **train_models.sh** to train the VAEs 
  
- Plots <br />
  Set the **output_root_dir** in [codebase/plots/icml_plots.py](codebase/plots/icml_plots.py) to be the same as the **output_root_dir** in [codebase/hparams/defaults.py](codebase/hparams/defaults.py) which was set in the beginning.
  Then run **icml_plots.sh**.
 

## Test your own generative models. 
The codebase is also modularized for testing your own decoder-based generative models. You need to register your model under [codebase/models/user_models](codebase/models/user_models), and register the Hparam object at [codebase/hparams/user_models](codebase/models/user_models). Your model should come with its decoder variance model.x_logvar as a scalar or vector tensor. Set **specific_model_path** to the path of your decoder weights.

### PyTorch: 
If the generative models is trained in PyTorch, the checkpoint should contain the key "state_dict" as the weights of the model.

### Others:
If the generative models is trained in other framewords, you'll need to manuually bridge and load the weights. For example, the AAEs were trained in tensorflow, with the weights saved as numpy, and then loaded as nn.Parameter in PyTorch. Refer to[codebase/utils/aad_utils](codebase/utils/aad_utils) for more details.
  

## Detailed Experimental Settings
More details on how to control experimental settings can be found below. 


General configuration:

- `specific_model_path`: (str) Set to the path to the decoder weights for your own experiments. Set to None if you are reproducing our experiments.
- `original_experiment`: (boolean) This should be set to **True** when the checkpoint or the model is from the paper. When you are testing your own generative model, set this False and it will load from `specific_model_path` instead of the directories generated by this codebase. You may need to custimize the **load_user_model** function in [codebase/utils/experiment_utils.py](codebase/utils/experiment_utils.py) for your own generative model.   
- `output_root_dir`: (str) The root dir for the experiment workspace. Experiment results, checkpoints will be saved in subfolders under this directory.
- `group_list`: (list) A list specifying the file tree structure for the output of this experiment, inside the `output_root_dir`.   
- `step_sizes_target`: (str) When not defined or set to None, HMC step sizes adaptively tunned and saved during AIS. 
  When specifed as the name of hparam_set of another previously finished experiment, 
  HMC step sizez will be loaded from that experiment.
- `train_first`: (boolean) If set to true, experiment will first run training algorithm, and then run RD evaluation. Otherwise only RD evaluation will be run. 
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
  - `data_name`: (str) The name of the dataset to be used.
  - `train_loader`: (str) The name of the training loader to be used. 
  - `eval_train_loader`: (str) The loader for "train" in the **rd_data_list**. 
  - `eval_test_loader`: (str) The loader for "test" in the **rd_data_list**. 
  - `input_dims`: (list) A list specifying the input dimensions. 
  - `input_vector_length`: (int) The product of **input_dims**.
  

The rest: normally the below settings do not need to be changed.
  - `cuda`: (boolean) Whether or not to use CUDA. 
  - `verbose`: (boolean) Verbose or not for logging and print statements. 
  - `random_seed`: (int) Random seed.  
  - `n_test_batch`: (int) Number of batch you want to test on during training or IWAE. During training it'll test on a held-out validation set. 
 

