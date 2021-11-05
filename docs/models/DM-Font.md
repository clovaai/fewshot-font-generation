# Train
## 1. Modify the configuration file ("cfgs/DM/train.yaml")

You can set these values by giving command-line arguments like argparse, not modifying this configuration file directly.
For the detailed description, please refer [here](https://github.com/khanrc/sconf#cli-modification).

---
- **use_ddp**:  Whether to use DataDistributedParallel. Set True to this to use multi-gpu.
- **port**: The port for the DataDistributedParallel training.

- **decomposition**: The location of decomposition rule file.
- **n_primals**: The number of the entire primals.

- **trainer**: _(leave blank)_
  - **resume**: Path to the checkpoint to resume from.
  - **work_dir**: Path to save the checkpoints, the validation images, and log.
---

## 2. Run training

```
python train_DM.py cfgs/DM/train.yaml cfgs/data/train/custom.yaml --work_dir(optional) path/to/save/outputs
```
-g, -n, -nr, -p are arguments for the DistributedDataParallel training.
You do not need to give these arguments if you are using a single GPU.

* **arguments**
  * path/to/config (first argument): path to configration file.
    * Multiple values are allowed but the first one should locate in `cfgs/DM`.
  * \-g : number of gpus to use for the training.
  * \-n : number of nodes to use for the training.
  * \-nr : the ranking of current node within the nodes.
  * \-p : the port to use for the DistributedDataParallel training.
  * \-\-work_dir : path to save outputs. The `trainer.work_dir` in the configuration file will be overwrited to this value.
  

---

# Evaluate
## 1. Modify the configuration file ("cfgs/DM/eval.yaml")

All the arguments should be identical to the arguments used for the training the weight to evaluate.

---
- **decomposition**: The location of decomposition rule file.
- **n_primals**: The number of the entire primals.
---

## 2. Run evaluation

```
python inference.py cfgs/DM/eval.yaml cfgs/data/eval/kor_ttf.yaml \
--model DM \
--weight weights/DM_kor.pth \
--result_dir ./result/DM
```
* **arguments**
  * path/to/config (first argument, multiple values are allowed): path to configration file.
    * Multiple values are allowed but the first one should locate in `cfgs/DM`.
  * \-\-model : The model to evaluate. DM, LF, MX and FUNIT are available.
  * \-\-weight: The weight to evaluate.
  * \-\-result_dir: Path to save generated images.