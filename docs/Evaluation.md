If you are using FTransGAN Dataset, see Evaluating section of [here](docs/FTransGAN-Dataset.md).


# Train
## 1. Modify the configuration file ("cfgs/evaluator/train.yaml")

You can set these values by giving command-line arguments like argparse, not modifying this configuration file directly.
For the detailed description, please refer [here](https://github.com/khanrc/sconf#cli-modification).

---
- **trainer**: _(leave blank)_
  - **resume**: Path to the checkpoint to resume from.
  - **work_dir**: Path to save the checkpoints, the validation images, and log.
  - **max_epoch**: Epochs to train the model.
  
- **dset**: _(leave blank)_
  - **train**: _(leave blank)_
    - **data_dir**: Path to the data to use for the training.
      - List format is allowed: the training data will be collected from all the paths in this list.
    - **chars**: The character list to train the model to classify.
    - **extension**: The extesion of training data.
    - **save_list**: Whether to save the list of fonts and chars. 
      - The list of fonts and chars will be saved to `trainer.work_dir`.
      - You may need the list of fonts and characters for the evaluation.
  - **val**: _(leave blank)_
    - **n_val_example**: The number of data to validate.
    - **data_dir**: Path to the data to use for the validation.
    - **extension**: The extesion of validation data.

---

## 2. Run training

```
python train_evaluator.py cfgs/evaluator/train.yaml -g(optional) 2 -n(optional) 2 -nr(optional) -p(optional) 12241 0 --work_dir(optional) path/to/save/outputs
```
-g, -n, -nr, -p are arguments for the DistributedDataParallel training.
You do not need to give these arguments if you are using a single GPU.

* **arguments**
  * path/to/config (first argument): path to configration file.
    * Multiple values are allowed but the first one should locate in `cfgs/evaluator`.
  * \-g : number of gpus to use for the training.
  * \-n : number of nodes to use for the training.
  * \-nr : the ranking of current node within the nodes.
  * \-p : the port to use for the DistributedDataParallel training.
  * \-\-work_dir : path to save outputs. The `trainer.work_dir` in the configuration file will be overwrited to this value.
  

---

# Evaluate

## 1. Modify the configuration file ("cfgs/evaluator/eval.yaml")

You can set these values by giving command-line arguments like argparse, not modifying this configuration file directly.
For the detailed description, please refer [here](https://github.com/khanrc/sconf#cli-modification).

---
- **style_model_path**: The checkpoint file which contains the weight of style classifier.
- **content_model_path**: The checkpoint file which contains the weight of content(character) classifier.

- **dset**: _(leave blank)_
  - **test**: _(leave blank)_
    - **data_dir**: Path to the generated images.
    - **gt_dir**: Path to the ground truth data (to calculate SSIM and LPIPS).
    - **gt_extension**: The extension of ground truth data.
    - **keylist**: The list of fonts which used to train the evaluator. You can obtain this by setting `dset.train.save_list` to `True` when running the training.
    - **charlist**: The list of characters which used to train the evaluator. You can obtain this by setting `dset.train.save_list` to `True` when running the training.
    

## 2. Run evaluation

```
python eval.py cfgs/evaluator/eval.yaml \
--result_dir path/to/save/result/file \
--result_name eval
```
* **arguments**
  * path/to/config (first argument, multiple values are allowed): path to configration file.
  * \-\-result_dir: Path to save result json file.
  * \-\-n_ref: Name of the result json file. (Not need to contain ".json" - it will be added automatically.)