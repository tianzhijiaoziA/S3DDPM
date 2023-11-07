# S3-DDPM

**Paper**: Self-fusion Simplex Noise-based Diffusion Model for Self-supervised Low-dose Digital Radiography Denoising

**Authors**: Yanyang Wang, Zirong Li, Jianjia Zhang and Weiwen Wu

Date : 11-07-2023  
Version : 1.0

## Requirements and Dependencies
``` bash
pip install -r requirements.txt
```

## Training
``` bash
To train a model, run `python3 diffusion_training.py ARG_NUM` where `ARG_NUM` is the number relating to the json arg
file. These arguments are stored in ./test_args/ and are called args1.json for example.
```

## Test Demo
``` bash
python evaluation.py ARG_NUM
```