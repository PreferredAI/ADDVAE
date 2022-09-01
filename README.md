## Introduction
 - This is the implementation of paper 
> Nhu-Thuat Tran and Hady W. Lauw (2022). Aligning Dual Disentangled User Representations from Ratings and Textual Content. In Proceedings of 28th ACM SIGKDD, Washington, DC, August 14-18, 2022.

## Dataset
 - Prepare your own dataset as sample and place under `data` folder

## Create virtual environment
 - Python version: ```3.7.5```
 - Create your project environment with `virtualenv`: ```virtualenv env --python=path_to_your_python``` (remove ```--python``` option if your default Python version is ```3.7.5```).
 - Install required libraries in `requirements.txt`.

## Run the code
 - Create a bash file `run.sh`
 - Run command ```bash run.sh```

## Citation
If you want to use our codes in your research, please cite:
```
@inproceedings{ADDVAE:2022,
    author = {Tran, Nhu-Thuat and Lauw, Hady W.},
    title = {Aligning Dual Disentangled User Representations from 
             Ratings and Textual Content},
    year = {2022},
    booktitle = {Proceedings of the 28th ACM SIGKDD Conference on 
                 Knowledge Discovery and Data Mining},
    pages = {1798–1806}
    location = {Washington DC, USA}
}
```