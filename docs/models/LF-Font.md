# Train
## Modify the configuration file ("cfgs/LF/p1/train.yaml", "cfgs/LF/p2/train.yaml")

You can set these values by giving command-line arguments like argparse, not modifying this configuration file directly.
For the detailed description, please refer [here](https://github.com/khanrc/sconf#cli-modification).

---
- **use_ddp**:  Whether to use DataDistributedParallel. Set True to this to use multi-gpu.
- **port**: The port for the DataDistributedParallel training.

- **decomposition**: The location of decomposition rule file.
- **primals**: The location of primals list file.

- **trainer**: _(leave blank)_
  - **resume**: Path to the checkpoint to resume from.
  - **work_dir**: Path to save the checkpoints, the validation images, and log.
  
- **dset**: _(leave blank)_
  - **train**: (leave blank)
    - **source_path** : path to the source font or source directory to use for the validation.
    - **source_ext**: extension of the source data. 
        - If you are using a ttf file, set this to "ttf".
        - If you are using image files, set this to their extension ("png", "jpg" ...).
    
- **gen** _(leave blank)_ _(only in "cfgs/LF/p2/train.yaml")_
  - **emb_dim**: the dimension of style and component factors. _(only in "cfgs/LF/p2/train.yaml")_

---

## Phase 1 training

```
python train_LF.py cfgs/LF/p1/train.yaml cfgs/data/train/custom.yaml --phase 1 --work_dir(optional) path/to/save/outputs
```
-g, -n, -nr, -p are arguments for the DistributedDataParallel training.
You do not need to give these arguments if you are using a single GPU.

* **arguments**
  * path/to/config (first argument): path to configration file.
    * Multiple values are allowed but the first one should locate in `cfgs/LF`.
  * \-g : number of gpus to use for the training.
  * \-n : number of nodes to use for the training.
  * \-nr : the ranking of current node within the nodes.
  * \-p : the port to use for the DistributedDataParallel training.
  * \-\-work_dir : path to save outputs. The `trainer.work_dir` in the configuration file will be overwrited to this value.
  

---
## Phase 2 training

```
python train_LF.py cfgs/LF/p2/train.yaml cfgs/data/train/custom.yaml --resume path/to/phase1/weights --phase 2 --work_dir(optional) path/to/save/outputs
```
* **arguments**
  * path/to/config (first argument, multiple values are allowed): path to configration file.
    * Multiple values are allowed but the first one should locate in `cfgs/LF/p2`.
  * \-\-resume : path to the weight which saved by phase 1 training.
  * \-\-phase : The training phase. 1 or 2 is available.
  * \-\-work_dir(optional) : path to save outputs. The `trainer.work_dir` in the configuration file will be overwrited to this value.

---

# Evaluate
## 1. Modify the configuration file ("cfgs/LF/p2/eval.yaml")

All the arguments should be identical to the arguments used for the training the weight to evaluate.

---
- **decomposition**: The location of decomposition rule file.
- **primals**: The location of primals list file.
---

## 2. Run evaluation

```
python inference.py cfgs/LF/p2/eval.yaml cfgs/data/eval/chn_ttf.yaml \
--model LF \
--weight weights/LF_chn.pth \
--result_dir ./result/LF
```
* **arguments**
  * path/to/config (first argument): path to configration files.
    * Multiple values are allowed but the first one should locate in `cfgs/LF/p2`.
  * \-\-model : The model to evaluate. DM, LF, MX and FUNIT are available.
  * \-\-weight: The weight to evaluate.
  * \-\-result_dir: Path to save generated images.
  * \-\-n_ref: The number of reference characters to use for the generation.
