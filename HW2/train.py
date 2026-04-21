"""
DINO training loop – single GPU, COCO evaluation, W&B logging.
"""
import contextlib
import copy
import datetime
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from config import get_config
from dataset import build_dataset, get_coco_api
from model import build_model

try:
    import wandb
except ImportError:
    wandb = None

# ── inlined data-loading helper ─────────────────────────────────────────────


def _collate(batch):
    """Collate variable-size images into an ImageBatch + targets list."""
    from model import ImageBatch  # avoid circular import at module level
    imgs, tgts = zip(*batch)
    imgs = list(imgs)
    mh = max(im.shape[1] for im in imgs)
    mw = max(im.shape[2] for im in imgs)
    stacked = imgs[0].new_zeros(len(imgs), 3, mh, mw)
    pmask = torch.ones(len(imgs),
                       mh,
                       mw,
                       dtype=torch.bool,
                       device=imgs[0].device)
    for k, im in enumerate(imgs):
        _, h, w = im.shape
        stacked[k, :, :h, :w] = im
        pmask[k, :h, :w] = False
    return ImageBatch(stacked, pmask), list(tgts)


# ── smoothed metric tracking ────────────────────────────────────────────────


class _Meter:
    """Running window average for a single scalar."""

    def __init__(self, win=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self._d = deque(maxlen=win)
        self._sum = 0.
        self._n = 0
        self._fmt = fmt

    def update(self, v, n=1):
        self._d.append(v)
        self._sum += v * n
        self._n += n

    def sync(self):
        pass  # single GPU

    @property
    def median(self):
        return float(torch.tensor(list(self._d)).median())

    @property
    def global_avg(self):
        return self._sum / max(self._n, 1)

    @property
    def value(self):
        return self._d[-1] if self._d else 0.

    def __str__(self):
        return self._fmt.format(median=self.median,
                                global_avg=self.global_avg,
                                max=max(self._d) if self._d else 0,
                                value=self.value)


class _MetricBoard:
    """Lightweight multi-meter tracker."""

    def __init__(self, sep="  "):
        self._meters = defaultdict(lambda: _Meter())
        self._sep = sep

    def put(self, **kw):
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self._meters[k].update(v)

    def add(self, name, meter):
        self._meters[name] = meter

    def sync(self):
        for m in self._meters.values():
            m.sync()

    @property
    def meters(self):
        return self._meters

    def __str__(self):
        return self._sep.join(f"{k}: {m}" for k, m in self._meters.items())


def _reduce(d):
    """No-op (single GPU)."""
    return d


# ── COCO evaluation helpers ─────────────────────────────────────────────────


class CocoEvalWrapper:
    """Accumulates per-image predictions and runs COCOeval."""

    def __init__(self, gt, tasks=("bbox",)):
        self._gt = copy.deepcopy(gt)
        self._tasks = list(tasks)
        self._evals = {t: COCOeval(self._gt, iouType=t) for t in tasks}
        self._img_ids = []
        self._per_task = {t: [] for t in tasks}

    def add(self, preds):
        ids = list(np.unique(list(preds.keys())))
        self._img_ids.extend(ids)
        for t in self._tasks:
            dets = self._to_coco(preds, t)
            with open(os.devnull, "w") as null:
                with contextlib.redirect_stdout(null):
                    dt = COCO.loadRes(self._gt, dets) if dets else COCO()
            ev = self._evals[t]
            ev.cocoDt = dt
            ev.params.imgIds = list(ids)
            _, ei = _exec_eval(ev)
            self._per_task[t].append(ei)

    def finalise(self):
        for t in self._tasks:
            arr = np.concatenate(self._per_task[t], 2)
            ev = self._evals[t]
            ev.evalImgs = list(arr.flatten())
            ev.params.imgIds = list(np.unique(self._img_ids))
            ev._paramsEval = copy.deepcopy(ev.params)
        for ev in self._evals.values():
            ev.accumulate()

    def report(self):
        for t, ev in self._evals.items():
            print(f"IoU metric: {t}")
            ev.summarize()

    def _to_coco(self, preds, iou_type):
        out = []
        for img_id, p in preds.items():
            if len(p) == 0:
                continue
            bx = p["boxes"]
            xywh = torch.stack(
                [bx[:, 0], bx[:, 1], bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]],
                1)
            for k in range(len(p["scores"])):
                out.append({
                    "image_id": img_id,
                    "category_id": p["labels"][k].item(),
                    "bbox": xywh[k].tolist(),
                    "score": p["scores"][k].item()
                })
        return out


def _exec_eval(ev):
    p = ev.params
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    ev.params = p
    ev._prepare()
    cats = p.catIds if p.useCats else [-1]
    ev.ious = {(i, c): ev.computeIoU(i, c) for i in p.imgIds for c in cats}
    md = p.maxDets[-1]
    imgs = [
        ev.evaluateImg(i, c, a, md)
        for c in cats
        for a in p.areaRng
        for i in p.imgIds
    ]
    imgs = np.asarray(imgs).reshape(len(cats), len(p.areaRng), len(p.imgIds))
    ev._paramsEval = copy.deepcopy(p)
    return p.imgIds, imgs


def _map_at_iou(evaluator, iou_thr=0.95):
    """Extract mAP at a specific IoU threshold from a finished evaluator."""
    if evaluator is None or "bbox" not in evaluator._evals:
        return float("nan")
    ev = evaluator._evals["bbox"]
    if "precision" not in ev.eval:
        return float("nan")
    idx = np.where(np.isclose(ev.params.iouThrs, iou_thr))[0]
    if len(idx) == 0:
        return float("nan")
    p = ev.eval["precision"][idx[0], :, :, 0, -1]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float("nan")


# ── W&B ─────────────────────────────────────────────────────────────────────


def _init_wandb(cfg):
    env_path = Path(".env")
    if env_path.exists():
        for ln in env_path.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    key = os.getenv("WANDB_API_KEY") or os.getenv("wandb_api_key")
    if not key or wandb is None:
        return None
    os.environ["WANDB_API_KEY"] = key
    return wandb.init(project=cfg.get("wandb_project", "dino-model"),
                      config=cfg,
                      name=cfg.get("wandb_run_name"),
                      reinit=True)


# ── optimiser (3 param groups) ──────────────────────────────────────────────


def _make_optimizer(model, cfg):
    lr = cfg["lr"]
    lr_bb = cfg["lr_backbone"]
    lp_mul = cfg.get("lr_linear_proj_mult", .1)
    lp_keys = cfg.get("lr_linear_proj_names",
                      ["reference_points", "sampling_offsets"])
    wd = cfg["weight_decay"]

    grp_bb, grp_lp, grp_rest = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            grp_bb.append(p)
        elif any(k in name for k in lp_keys):
            grp_lp.append(p)
        else:
            grp_rest.append(p)

    return torch.optim.AdamW([
        {
            "params": grp_rest,
            "lr": lr
        },
        {
            "params": grp_bb,
            "lr": lr_bb
        },
        {
            "params": grp_lp,
            "lr": lr * lp_mul
        },
    ],
                             lr=lr,
                             weight_decay=wd)


# ── one epoch ───────────────────────────────────────────────────────────────


def run_train_epoch(model,
                    criterion,
                    loader,
                    optim,
                    device,
                    epoch,
                    clip=0.,
                    use_dn=True):
    model.train()
    criterion.train()
    board = _MetricBoard()
    board.add("lr", _Meter(win=1, fmt="{value:.6f}"))
    board.add("class_error", _Meter(win=1, fmt="{value:.2f}"))

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        out = model(samples, targets) if use_dn else model(samples)
        ld = criterion(out, targets)
        wd = criterion.weight_dict
        loss = sum(ld[k] * wd[k] for k in ld if k in wd)

        rd = _reduce(ld)
        scaled = {k: v * wd[k] for k, v in rd.items() if k in wd}
        loss_val = sum(scaled.values()).item()
        if not math.isfinite(loss_val):
            print(f"Loss={loss_val}, aborting")
            sys.exit(1)

        optim.zero_grad()
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()

        unscaled = {f"{k}_unscaled": v for k, v in rd.items()}
        board.put(loss=loss_val, **scaled, **unscaled)
        if "class_error" in rd:
            board.put(class_error=rd["class_error"])
        board.put(lr=optim.param_groups[0]["lr"])
        pbar.set_postfix(loss=f"{loss_val:.4f}",
                         lr=f"{optim.param_groups[0]['lr']:.2e}")

    board.sync()
    return {k: m.global_avg for k, m in board.meters.items()}


@torch.no_grad()
def run_eval_epoch(model,
                   criterion,
                   post,
                   loader,
                   gt_coco,
                   device,
                   epoch,
                   use_dn=True):
    model.eval()
    criterion.eval()
    board = _MetricBoard()
    board.add("class_error", _Meter(win=1, fmt="{value:.2f}"))
    evaluator = CocoEvalWrapper(gt_coco, ["bbox"])

    for samples, targets in loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        out = model(samples, targets) if use_dn else model(samples)
        ld = criterion(out, targets)
        wd = criterion.weight_dict
        rd = _reduce(ld)
        scaled = {k: v * wd[k] for k, v in rd.items() if k in wd}
        unscaled = {f"{k}_unscaled": v for k, v in rd.items()}
        vloss = sum(scaled.values()).item()
        board.put(loss=vloss, **scaled, **unscaled)
        if "class_error" in rd:
            board.put(class_error=rd["class_error"])

        sizes = torch.stack([t["orig_size"] for t in targets])
        results = post["bbox"](out, sizes)
        evaluator.add({
            t["image_id"].item(): r for t, r in zip(targets, results)
        })

    board.sync()
    evaluator.finalise()
    with open(os.devnull,
              "w") as _devnull, contextlib.redirect_stdout(_devnull):
        evaluator.report()

    stats = {k: m.global_avg for k, m in board.meters.items()}
    stats["coco_eval_bbox"] = evaluator._evals["bbox"].stats.tolist()
    return stats, evaluator


# ── main ────────────────────────────────────────────────────────────────────


def main():
    cfg = get_config()
    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    wb = _init_wandb(cfg)

    model, criterion, post = build_model(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optim = _make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                       T_max=cfg["epochs"],
                                                       eta_min=cfg.get(
                                                           "lr_eta_min", 1e-6))

    with open(os.devnull,
              "w") as _devnull, contextlib.redirect_stdout(_devnull):
        ds_tr = build_dataset("train", cfg)
        ds_va = build_dataset("val", cfg)
    dl_tr = DataLoader(ds_tr,
                       batch_sampler=torch.utils.data.BatchSampler(
                           torch.utils.data.RandomSampler(ds_tr),
                           cfg["batch_size"],
                           drop_last=True),
                       collate_fn=_collate,
                       num_workers=cfg["num_workers"])
    dl_va = DataLoader(ds_va,
                       cfg["batch_size"],
                       sampler=torch.utils.data.SequentialSampler(ds_va),
                       drop_last=False,
                       collate_fn=_collate,
                       num_workers=cfg["num_workers"])
    base_ds = get_coco_api(ds_va)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg["output_dir"]) / stamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out}")

    tb_writer = SummaryWriter(
        log_dir=str(Path(cfg["output_dir"]) / "logs" / stamp))

    start_ep = 0
    dn = cfg.get("use_dn", True)

    if cfg.get("resume") and os.path.isfile(cfg["resume"]):
        print(f"Resuming from {cfg['resume']}")
        ckpt = torch.load(cfg["resume"], map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["lr_scheduler"])
            start_ep = ckpt["epoch"] + 1

    best_map, best_vloss = 0., float("inf")
    t0 = time.time()

    for epoch in range(start_ep, cfg["epochs"]):
        ep_t0 = time.time()
        tr = run_train_epoch(model, criterion, dl_tr, optim, device, epoch,
                             cfg["clip_max_norm"], dn)
        sched.step()

        state = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "lr_scheduler": sched.state_dict(),
            "epoch": epoch,
            "config": cfg
        }
        ckpt_paths = [out / "checkpoint.pth"]
        lr_drop = cfg.get("lr_drop", 999)
        if (epoch + 1) % lr_drop == 0 or (epoch + 1) % 100 == 0:
            ckpt_paths.append(out / f"checkpoint{epoch:04}.pth")
        for cp in ckpt_paths:
            torch.save(state, cp)

        va, evaluator = run_eval_epoch(model, criterion, post, dl_va, base_ds,
                                       device, epoch, dn)
        cs = va.get("coco_eval_bbox", [float("nan")] * 12)
        mAP = float(cs[0]) if len(cs) > 0 else float("nan")
        mAP50 = float(cs[1]) if len(cs) > 1 else float("nan")
        mAP75 = float(cs[2]) if len(cs) > 2 else float("nan")
        mAP95 = _map_at_iou(evaluator, 0.95)
        vloss = float(va.get("loss", float("inf")))
        ep_sec = int(time.time() - ep_t0)

        tr_acc = 1.0 - float(tr.get("class_error", 100.0)) / 100.0
        va_acc = 1.0 - float(va.get("class_error", 100.0)) / 100.0
        va_cls = float(va.get("loss_ce", float("nan")))
        va_bbox = float(va.get("loss_bbox", float("nan")))
        va_giou = float(va.get("loss_giou", float("nan")))

        print(
            f"  Epoch {epoch+1:03d}  train_loss={float(tr.get('loss',float('nan'))):.4f}  "
            f"val_loss={vloss:.4f}  mAP={mAP:.4f}  mAP@50={mAP50:.4f}  "
            f"mAP@75={mAP75:.4f}  mAP@95={mAP95:.4f}  "
            f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  ({ep_sec}s)  "
            f"cls={va_cls:.4f}  bbox={va_bbox:.4f}  giou={va_giou:.4f}")

        saved = False
        if vloss < best_vloss:
            best_vloss = vloss
            saved = True
        if mAP > best_map:
            best_map = mAP
            saved = True
        if saved:
            torch.save(state, out / "best.pth")
            print(
                f"  -> Best model saved (mAP={mAP:.4f}, val_loss={vloss:.4f})")

        if wb is not None:
            wandb.log(
                {
                    "epoch":
                        epoch,
                    "train/loss":
                        float(tr.get("loss", float("nan"))),
                    "train/loss_ce":
                        float(tr.get("loss_ce", float("nan"))),
                    "train/loss_bbox":
                        float(tr.get("loss_bbox", float("nan"))),
                    "train/loss_giou":
                        float(tr.get("loss_giou", float("nan"))),
                    "train/class_error":
                        float(tr.get("class_error", float("nan"))),
                    "train/lr":
                        float(tr.get("lr", optim.param_groups[0]["lr"])),
                    "valid/loss":
                        vloss,
                    "valid/loss_ce":
                        float(va.get("loss_ce", float("nan"))),
                    "valid/loss_bbox":
                        float(va.get("loss_bbox", float("nan"))),
                    "valid/loss_giou":
                        float(va.get("loss_giou", float("nan"))),
                    "valid/class_error":
                        float(va.get("class_error", float("nan"))),
                    "valid/mAP@50":
                        mAP50,
                    "valid/mAP@75":
                        mAP75,
                    "valid/mAP@95":
                        mAP95,
                    "valid/mAP@50:95":
                        mAP,
                },
                step=epoch)

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss",
                                 float(tr.get("loss", float("nan"))), epoch)
            tb_writer.add_scalar("train/loss_ce",
                                 float(tr.get("loss_ce", float("nan"))), epoch)
            tb_writer.add_scalar("train/loss_bbox",
                                 float(tr.get("loss_bbox", float("nan"))),
                                 epoch)
            tb_writer.add_scalar("train/loss_giou",
                                 float(tr.get("loss_giou", float("nan"))),
                                 epoch)
            tb_writer.add_scalar("train/class_error",
                                 float(tr.get("class_error", float("nan"))),
                                 epoch)
            tb_writer.add_scalar(
                "train/lr", float(tr.get("lr", optim.param_groups[0]["lr"])),
                epoch)

            tb_writer.add_scalar("valid/loss", vloss, epoch)
            tb_writer.add_scalar("valid/loss_ce",
                                 float(va.get("loss_ce", float("nan"))), epoch)
            tb_writer.add_scalar("valid/loss_bbox",
                                 float(va.get("loss_bbox", float("nan"))),
                                 epoch)
            tb_writer.add_scalar("valid/loss_giou",
                                 float(va.get("loss_giou", float("nan"))),
                                 epoch)
            tb_writer.add_scalar("valid/class_error",
                                 float(va.get("class_error", float("nan"))),
                                 epoch)

            tb_writer.add_scalar("valid/mAP@50", mAP50, epoch)
            tb_writer.add_scalar("valid/mAP@75", mAP75, epoch)
            tb_writer.add_scalar("valid/mAP@95", mAP95, epoch)
            tb_writer.add_scalar("valid/mAP@50:95", mAP, epoch)

        with (out / "log.txt").open("a") as f:
            f.write(
                json.dumps({
                    **{
                        f"train_{k}": v for k, v in tr.items()
                    },
                    **{
                        f"test_{k}": v for k, v in va.items()
                    },
                    "epoch": epoch,
                    "n_parameters": n_params,
                    "coco_mAP@50": mAP50,
                    "coco_mAP@75": mAP75,
                    "coco_mAP@95": mAP95,
                    "coco_mAP@50:95": mAP,
                }) + "\n")

    if wb is not None:
        wandb.finish()

    if tb_writer is not None:
        tb_writer.close()

    print(f"Done in {datetime.timedelta(seconds=int(time.time() - t0))}")


if __name__ == "__main__":
    main()
