# Instruction

The examples of datasets are in *(./data_example)*
Note that, we only provide the example font files; not the font files used for the training the provided weight *(generator.pth)*.
The example font files are downloaded from [here(chinese)](https://www.freechinesefont.com/tag/commercial-use-ok/) and [here(korean)](https://uhbeefont.com).

* This code can treat both dataset in image files(.png, .jpg ...) and truetype font files(.ttf).
* To use ttf data, see **1.1**.
* To use image data, see **1.2**.

## 1. Prepare dataset
### 1.1. Prepare TTF files
* Prepare the TrueType font files(.ttf) to use for the training and the validation.
* Put the training font files and validation font files into separate directories.
* In this case, the available characters list is needed in txt format. (described in **1.1.a**.)
* If you want to split characters to seen characters and unseen characters, the character split information is needed in json format. (described in **2**.)

    ---
    ### 1.1.a. Preparing the available characters list
    * If you have the available character list of a .ttf file, save its available characters list to a text file (.txt) with the same name in the same directory with the ttf file.
        * (example) 
        * **TTF file**: data/ttfs/train/MaShanZheng-Regular.ttf
        * **its available characters**: data/ttfs/train/MaShanZheng-Regular.txt
    * You can also generate the available characters files automatically using the get_chars_from_ttf.py
    * How to use:
        * `python get_chars_from_ttf.py path/to/ttf/dir`
        * **path/to/ttf/dir**: The root directory to find the .ttf files. All the .ttf files under this directory and its subdirectories will be processed.
    ---

### 1.2. Prepare Image files
* The images are should be placed in this format:
```
    * data_dir
    |-- font1
        |-- char1.png
        |-- char2.png
        |-- char3.png
    |-- font2
        |-- char1.png
        |-- char2.png
            .
            .
            .
```
* You can see the example at `data_example/chn/png`.
* The images with the same style are should be grouped with the same directory.
* The name of each image file should be its character.

## 2. Split characters to train and validation set 
(*required for 1.1, optional for 1.2*)
* Save the list of characters to use for the training as a json file.
    * This step can be skipped if you want to use all the available characters without splitting for the training.
    * The characters **both** existing in this list and available from the dataset will be used for the training.
    * Our example is in `data/chn/train_chars.json`(Chinese) and `data/kor/train_chars.json`(Korean)
* Save the list of characters to use for the validation as a json file.
    * The characters **both** existing in this list and available from the dataset will be used for the validation.
    * The characters that we used for the training is in `data/chn/val_unseen_chars.json`(Chinese) and `data/kor/val_unseen_chars.json`(Korean)

## 3. Prepare decomposition information 
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

## 4. Prepare source font or source images
(*not required for DM-Font*)
### 4.1 Using ttf file as source
    * You can get the Chinese source font which we used from [here](https://chinesefontdesign.com/font-housekeeper-song-ming-typeface-chinese-font-simplified-chinese-fonts.html).
      * Download this file and save to `data/chn/source.ttf`.
    * The Korean source font is attached at `data/kor/source.ttf`
### 4.2 Using image files as source
    * Put all the source image files in a single directory. 
    * The name of each image file should be its character.


## 5. Modify the data configuration file (cfgs/data/train/custom.yaml)

Please do not modify the indentation, because the indentation rule is very important in these configuration files. 

---
- **dset**:  (leave blank)
  - **train**:  (leave blank)
    - **data_dir**: path to training data.
    - **extension**:  extension of training data. 
        - Set this to "ttf" if you are using ttf files.
        - If you are using image files set this to the image files' extension.
    - **chars**: The json file containing the training characters.
        - If this is blank, all the available characters will be used for the training.
        - If you generated the file in step **2**, set this to its location.
  - **val**: (leave blank)
    - **name**: Anything is okay; this will be the name of the validation set.
      - **data_dir** : path to validation data.
      - **extension**:  extension of validation data.
      - **n_gen**: How many characters to generate during validation. The characters will be randomly selected from characters.
      - **n_font** How many fonts to generate during validation. The fonts will be randomly selected from available fonts.
      - **chars**: The json file containing the entire validation characters.
        - If this is blank, the validation characters will be chosen from all the available characters.
      - **source_path** : path to the source font or source directory to use for the validation.
      - **source_ext**: extension of the source data. 
        - If you are using a ttf file, set this to "ttf".
        - If you are using image files, set this to their extension ("png", "jpg" ...).
---
