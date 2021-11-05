# Train
## 1. Modify the configuration file ("cfgs/MX/train.yaml")

You can set these values by giving command-line arguments like argparse, not modifying this configuration file directly.
For the detailed description, please refer [here](https://github.com/khanrc/sconf#cli-modification).

---
- **decomposition**: The location of decomposition rule file.
- **n_primals**: The number of the entire primals.

- **trainer**: _(leave blank)_
  - **resume**: Path to the checkpoint to resume from.
  - **work_dir**: Path to save the checkpoints, the validation images, and log.
  
- **gen**: _(leave blank)_
  - **n_experts**: number of the experts.

---

## 2. Run training

```
python train_MX.py cfgs/MX/train.yaml cfgs/data/train/custom.yaml -g(optional) 2 -n(optional) 2 -nr(optional) -p(optional) 12241 0 --work_dir(optional) path/to/save/outputs
```
-g, -n, -nr, -p are arguments for the DistributedDataParallel training.
You do not need to give these arguments if you are using a single GPU.

* **arguments**
  * path/to/config (first argument): path to configration file.
    * Multiple values are allowed but the first one should locate in `cfgs/MX`.
  * \-g : number of gpus to use for the training.
  * \-n : number of nodes to use for the training.
  * \-nr : the ranking of current node within the nodes.
  * \-p : the port to use for the DistributedDataParallel training.
  * \-\-work_dir : path to save outputs. The `trainer.work_dir` in the configuration file will be overwrited to this value.
  

---

# Evaluate
## Run evaluation

```
python inference.py cfgs/MX/eval.yaml cfgs/data/eval/kor_ttf.yaml \
--model MX \
--weight weights/MX_chn.pth \
--result_dir ./result/MX
```
* **arguments**
  * path/to/config (first argument, multiple values are allowed): path to configration file.
  * \-\-model : The model to evaluate. DM, LF, MX and FUNIT are available.
  * \-\-weight: The weight to evaluate.
  * \-\-result_dir: Path to save generated images.
  * \-\-n_ref: The number of reference characters to use for the generation.