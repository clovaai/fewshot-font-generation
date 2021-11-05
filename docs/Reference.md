# Instruction

The examples of datasets are in *(./data_example)*
The example font files are downloaded from [here(chinese)](https://www.freechinesefont.com/tag/commercial-use-ok/) and [here(korean)](https://uhbeefont.com).

* This code can treat both dataset in image files(.png, .jpg ...) and truetype font files(.ttf).
* To use ttf data, see **1.1**.
* To use image data, see **1.2**.

## 1. Prepare dataset (Same with training set)
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
        * `python get_chars_from_ttf.py --root_dir path/to/ttf/dir`
        * **--root_dir**: The root directory to find the .ttf files. All the .ttf files under this directory and its subdirectories will be processed.
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

## 2. Prepare source font or source images
(*not required for DM-Font*)
### 2.1 Using ttf file as source
    * You can get the Chinese source font which we used from [here](https://chinesefontdesign.com/font-housekeeper-song-ming-typeface-chinese-font-simplified-chinese-fonts.html).
      * Download this file and save to `data/chn/source.ttf`.
    * The Korean source font is attached at `data/kor/source.ttf`
### 2.2 Using image files as source
    * Put all the source image files in a single directory. 
    * The name of each image file should be its character.

## 3. Prepare the character files: reference characters and characters to generate
* Save the list of characters to use as the reference characters.
    * This step can be skipped if you want to use **all the available characters**.
    * The characters **both** existing in this list and available from the dataset will be used as the reference.
    * Our example is in `data/chn/ref_chars.json`(Chinese) and `data/kor/ref_chars.json`(Korean)
* Save the list of characters to generate as a json file.
    * You can skip this if you want to generate every characters in the source font.
    * The characters **both** existing in this list and available from source font will be used for the validation.
    * The characters that we used for the training is in `data/gen_chars.json`(Chinese) and `data/kor/gen_chars.json`(Korean)

## 4. Prepare decomposition information 
(*not required for MX-Font and FUNIT*)
* The files should be identical to those used to train the evaluating weight.

## 5. Modify the data configuration file (cfgs/data/eval/custom.yaml)

Please do not modify the indentation, because the indentation rule is very important in these configuration files.

* The files in `cfgs/data/eval` are the examples.

---
- **dset**:  (leave blank)
  - **test**:  (leave blank)
    - **extension**:  extension of training data. 
        - Set this to "ttf" if you are using ttf files.
        - If you are using image files set this to the image files' extension.
    - **data_dir**: path to training data.
    - **source_path** : path to the source font or source directory to use for the validation.
    - **source_ext**: extension of the source data. 
        - If you are using a ttf file, set this to "ttf".
        - If you are using image files, set this to their extension ("png", "jpg" ...).
    - **ref_chars**: The json file containing the reference characters.
        - If this is blank, all the available characters will be used as the reference.
    - **gen_chars**: The json file containing the characters list to generate.
        - If this is blank, all the available characters in the source font will be generated.
---
