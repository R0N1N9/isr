"""Create verifiers: zero-shot CLIP/SigLIP, or a crop classifier trained on your reviews.

Both need PyTorch at build time only (``pip install -e ".[train]"``). The
result is an ONNX file plus card that runs in the field with ONNX Runtime.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import Dataset
from .verifier import IMAGENET, VerifierCard, context_crop

DEFAULT_PROMPTS = {
    "person": {
        "positive": ["an aerial photo of a person", "an aerial photo of a person lying on the ground",
                     "a drone photo of a hiker seen from above", "an overhead photo of a person standing outdoors"],
        "negative": ["an aerial photo of a rock", "an aerial photo of a tree", "an aerial photo of a bush",
                     "an aerial photo of grass", "an aerial photo of a fallen log", "an aerial photo of a shadow",
                     "an aerial photo of bare ground", "an aerial photo of a road", "an aerial photo of water",
                     "an aerial photo of a car", "an aerial photo of litter"],
    },
    "vehicle": {
        "positive": ["an aerial photo of a car", "an aerial photo of a truck", "an aerial photo of a parked vehicle"],
        "negative": ["an aerial photo of a roof", "an aerial photo of a rock", "an aerial photo of a tree",
                     "an aerial photo of a road", "an aerial photo of grass", "an aerial photo of a shed"],
    },
    "animal": {
        "positive": ["an aerial photo of a dog", "an aerial photo of a horse", "an aerial photo of a cow",
                     "an aerial photo of an animal"],
        "negative": ["an aerial photo of a rock", "an aerial photo of a bush", "an aerial photo of a tree stump",
                     "an aerial photo of grass", "an aerial photo of a shadow", "an aerial photo of a person"],
    },
}


def build_clip_verifier(out_path: Path, model: str = "ViT-B-16-SigLIP", pretrained: str | None = "webli",
                        target: str = "person", prompts: dict | None = None, log=print) -> Path:
    """Export an OpenCLIP image encoder to ONNX and pre-compute prompt embeddings.

    Run once with internet access (weights download from Hugging Face). Use
    ``pretrained=None`` only for testing: random weights.
    """
    import open_clip
    import torch

    prompts = prompts or DEFAULT_PROMPTS[target]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Loading {model} ({pretrained or 'random weights'})...")
    m, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
    m.eval()
    tok = open_clip.get_tokenizer(model)
    with torch.no_grad():
        emb = {}
        for k in ("positive", "negative"):
            t = m.encode_text(tok(prompts[k])).float()
            emb[k] = (t / t.norm(dim=-1, keepdim=True)).numpy()
        scale = float(m.logit_scale.exp().item()) if hasattr(m, "logit_scale") else 100.0
    size = m.visual.image_size
    size = int(size[0] if isinstance(size, (tuple, list)) else size)
    mean, std = list(IMAGENET[0]), list(IMAGENET[1])
    for t in getattr(preprocess, "transforms", []):
        if t.__class__.__name__ == "Normalize":
            mean, std = [float(v) for v in t.mean], [float(v) for v in t.std]

    class Encoder(torch.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.m = mm

        def forward(self, x):
            return self.m.encode_image(x)

    dummy = torch.zeros(1, 3, size, size)
    log(f"Exporting image encoder to {out_path} ...")
    _export(Encoder(m).eval(), dummy, out_path, "embeddings")
    np.savez(out_path.with_suffix(".prompts.npz"), positive=emb["positive"], negative=emb["negative"])
    VerifierCard(type="embedding", input_size=size, mean=mean, std=std, prompts=prompts, logit_scale=scale,
                 target=target, notes=f"{model} / {pretrained or 'RANDOM WEIGHTS (test only)'}").save(out_path)
    return out_path


def _export(module, dummy, out_path: Path, output_name: str) -> None:
    import torch

    kw = dict(input_names=["images"], output_names=[output_name],
              dynamic_axes={"images": {0: "n"}, output_name: {0: "n"}}, opset_version=17)
    try:
        torch.onnx.export(module, dummy, str(out_path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(module, dummy, str(out_path), **kw)


# --------------------------------------------------------------------------
# crop classifier from your own reviews
# --------------------------------------------------------------------------


def _crops(ds: Dataset, split: str, context: float, rng: random.Random):
    """(PIL crop, class) pairs: every labelled box, every dismissed false alarm, plus random background."""
    out = []
    for s in ds.samples:
        if s.split != split:
            continue
        im = Image.open(ds.dir / s.file).convert("RGB")
        for b in s.boxes:
            out.append((context_crop(im, b["box"], context), b["cls"]))
        cb = s.source.get("cand_box")
        if not s.boxes and cb:
            out.append((context_crop(im, cb, context), "background"))
        for _ in range(1 if s.boxes else 2):  # random background patches clear of every box
            for _try in range(10):
                w = rng.uniform(12, 64)
                x, y = rng.uniform(0, s.width - w), rng.uniform(0, s.height - w)
                bx = [x, y, x + w, y + w]
                if not any(bx[0] < b["box"][2] and bx[2] > b["box"][0] and bx[1] < b["box"][3] and bx[3] > b["box"][1]
                           for b in s.boxes):
                    out.append((context_crop(im, bx, context), "background"))
                    break
    return out


def train_crop_classifier(dataset_dir: Path, out_path: Path, epochs: int = 15, size: int = 96,
                          weights: str = "imagenet", batch: int = 64, lr: float = 1e-3, context: float = 3.0,
                          seed: int = 0, log=print) -> dict:
    """Train MobileNetV3-Small to tell targets from false alarms; export to ONNX + card."""
    import torch
    import torchvision
    from torch import nn

    torch.manual_seed(seed)
    rng = random.Random(seed)
    ds = Dataset(dataset_dir)
    train, val = _crops(ds, "train", context, rng), _crops(ds, "val", context, rng)
    if not train:
        raise ValueError("dataset has no training samples")
    classes = ["background"] + sorted({c for _, c in train + val} - {"background"})
    idx = {c: i for i, c in enumerate(classes)}
    counts = {c: sum(1 for _, k in train if k == c) for c in classes}
    log(f"Training crops: {counts}; validation crops: {len(val)}")
    if counts["background"] == 0 or len(classes) < 2:
        raise ValueError("need both confirmed targets and dismissed/background examples to train a verifier")

    mean, std = torch.tensor(IMAGENET[0]).view(3, 1, 1), torch.tensor(IMAGENET[1]).view(3, 1, 1)

    def tensor(im: Image.Image, aug: bool):
        im = im.resize((size, size), Image.BICUBIC)
        if aug:  # top-down imagery has no "up": flips and 90-degree turns are free augmentation
            if rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im = im.rotate(90 * rng.randrange(4))
        x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)
        if aug:
            x = (x * rng.uniform(0.8, 1.2) + rng.uniform(-0.06, 0.06)).clamp(0, 1)
        return (x - mean) / std

    w_enum = None
    if weights == "imagenet":
        try:
            w_enum = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            net = torchvision.models.mobilenet_v3_small(weights=w_enum)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(classes))
            arch = "MobileNetV3-Small (ImageNet-pretrained)"
        except Exception as e:  # offline: no pretrained weights
            log(f"  ! couldn't load ImageNet weights ({e.__class__.__name__}); training a small network from scratch")
            w_enum = None
    if w_enum is None:
        # a big network from scratch on a few hundred crops overfits or collapses; a small one learns
        net = _tiny_net(len(classes))
        arch = "TinyNet (from scratch)"
    log(f"  architecture: {arch}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev)
    # class-balanced sampling: rare confirmed targets are seen as often as background
    wts = [1.0 / counts[c] for _, c in train]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        picks = rng.choices(range(len(train)), weights=wts, k=len(train))
        tot = 0.0
        for i in range(0, len(picks), batch):
            xb = torch.stack([tensor(train[j][0], True) for j in picks[i:i + batch]]).to(dev)
            yb = torch.tensor([idx[train[j][1]] for j in picks[i:i + batch]]).to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(yb)
        sched.step()
        log(f"  epoch {ep + 1}/{epochs}: loss {tot / len(picks):.3f}")

    _recalibrate_bn(net, [tensor(c, False) for c, _ in train], dev, batch)
    metrics = {"train_crops": len(train), "val_crops": len(val), "classes": classes,
               "seconds": round(time.time() - t0, 1), "pretrained": w_enum is not None, "architecture": arch}
    net.eval()
    if val:
        with torch.no_grad():
            pred = []
            for i in range(0, len(val), batch):
                xb = torch.stack([tensor(c, False) for c, _ in val[i:i + batch]]).to(dev)
                pred += net(xb).argmax(1).cpu().tolist()
        truth = [idx.get(c, 0) for _, c in val]
        pos = [i for i, t in enumerate(truth) if t != 0]
        neg = [i for i, t in enumerate(truth) if t == 0]
        metrics["val_accuracy"] = round(sum(p == t for p, t in zip(pred, truth)) / len(val), 3)
        metrics["val_target_recall"] = round(sum(pred[i] != 0 for i in pos) / len(pos), 3) if pos else None
        metrics["val_false_alarm_rejection"] = round(sum(pred[i] == 0 for i in neg) / len(neg), 3) if neg else None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _export(net.cpu(), torch.zeros(1, 3, size, size), out_path, "logits")
    target = classes[1] if len(classes) == 2 else "target"
    VerifierCard(type="classifier", input_size=size, mean=list(IMAGENET[0]), std=list(IMAGENET[1]),
                 classes=classes, positive=classes[1:], context=context, target=target,
                 notes=f"MobileNetV3-Small, {epochs} epochs, {len(train)} crops, "
                       f"{arch}; val {metrics}").save(out_path)
    return metrics


def _tiny_net(n_classes: int):
    from torch import nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))
    return nn.Sequential(block(3, 24), block(24, 48), block(48, 96), block(96, 160),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(160, n_classes))


def _recalibrate_bn(net, xs, dev, batch) -> None:
    """Recompute batch-norm statistics on clean training crops.

    Running averages accumulated during a short, heavily augmented training run
    can be far off, which makes the exported model output a constant. Exact
    statistics over the training set fix that.
    """
    import torch
    from torch import nn

    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns or not xs:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    net.train()
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            net(torch.stack(xs[i:i + batch]).to(dev))
