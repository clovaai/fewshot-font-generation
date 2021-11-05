# Training FFG Models

## 1. Download dataset
The dataset can be downloaded from [here](https://github.com/ligoudaner377/font_translator_gan#how-to-use).

## 2. Prepare decomposition information 
(*not required for FUNIT*)
* The files that we used are provided.
    * Chinese: data/chn/decomposition.json, data/chn/primals.json
    * Korean: data/kor/decomposition.json, data/kor/primals.json, data/kor/decomposition_DM.json(only for DM-Font)
* See **3.1** to build your own decomposition file:

    ---
    ### 3.1 The format of decomposition rule and primals file
    * **Decomposition rule**
        * structure: dict *(in json format)*
        * format: {char: [list of components]}
        * example: {'㐬': ['亠', '厶', '川'], '㐭': ['亠', '囗', '口']}
    * **Primals**
        * structure: list *(in json format)*
        * format: [**All** the components in the decomposition rule file]
        * example: ['亠', '厶', '川', '囗', '口']
    ---

## 3. Modify the data configuration file ("cfgs/data/train/chn_ftransgan.yaml")

### 1. copy `cfgs/data/train/chn_ftransgan.yaml` to your own configuration file.
Use this command: 
* `cp -f cfgs/data/train/chn_ftransgan.yaml cfgs/data/train/custom.yaml`

### 2. Modify the copied file ("cfgs/data/train/custom.yaml")
* Change the `(FTransGAN_root)` in the copied file to your own dataset root.

Please do not modify the indentation, because the indentation rule is very important in these configuration files. 


# Evaluating

We provide weights of the classifiers trained with FTransGAN Dataset. (weights/evaluator_ftransgan_20epoch.pth)
The list of labels are also provided in "data/chn/ftransgan/eval_keys.json" and "data/chn/ftransgan/eval_chars.json".

## 1. Modify the data configuration file ("cfgs/evaluator/eval_ftransgan.yaml")
1. Change the `dset.test.data_dir` to the root path of your generated images.
2. Change the `(FTransGAN_root)` in the copied file to your own dataset root.