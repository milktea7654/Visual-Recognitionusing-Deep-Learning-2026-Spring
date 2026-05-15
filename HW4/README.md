# HW4 Configurable PromptIR Framework

This version is organized around YAML configs and dynamic model loading.

## Structure

```text
train.py
inference.py
dataset.py
configs/
  train.yaml
  inference.yaml
  data.yaml
  promptir_cplfreq.yaml
models/
  __init__.py
  promptir_cplfreq.py
tools/
  check_pred_npz.py
```

Training outputs are saved as:

```text
outputs/
  YYYYMMDD_HHMMSS/
    checkpoints/
      best.pt
      last.pt
      epoch_0010.pt
    logs/
      events.out.tfevents.*
    config_resolved.yaml
```

## Train

```bash
python train.py --config configs/train.yaml
```

Override paths or settings from CLI:

```bash
python train.py \
  --config configs/train.yaml \
  --data_root /kaggle/input/hw4-realse-dataset/hw4_realse_dataset \
  --run_name promptir_cplfreq \
  --set train.epochs=400 \
  --set checkpoint.save_every=5 \
  --set model.dim=64
```

Open TensorBoard:

```bash
tensorboard --logdir outputs
```

## Inference

```bash
python inference.py \
  --config configs/inference.yaml \
  --data_root /kaggle/input/hw4-realse-dataset/hw4_realse_dataset \
  --checkpoint outputs/YYYYMMDD_HHMMSS/checkpoints/best.pt
```

Validate:

```bash
python tools/check_pred_npz.py outputs/pred.npz
```

## Add a new architecture

Create:

```text
models/my_model.py
configs/my_model.yaml
```

`models/my_model.py` must expose:

```python
def build_model(cfg):
    return MyModel(...)
```

Then set in the YAML:

```yaml
model:
  name: my_model
```

And train with:

```bash
python train.py --config configs/train.yaml --set model.name=my_model
```
