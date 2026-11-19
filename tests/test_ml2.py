"""Importer, scorecard, verifier and quantisation."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from isr.cli import main  # noqa: E402
from isr.ml.dataset import Dataset  # noqa: E402
from isr.ml.evaluate import SystemSpec, _summarise, evaluate, matches  # noqa: E402
from isr.ml.importers import detect_format, import_dataset  # noqa: E402
from isr.ml.verifier import Verifier, VerifierCard, apply_verification, context_crop  # noqa: E402
from isr.triage.detectors import Detection  # noqa: E402

from synth import draw_person, mavic3_gsd, terrain  # noqa: E402

G = mavic3_gsd(60)


def _scene(path: Path, seed: int):
    """1000x800 terrain with two people and a grey rock; returns the people's boxes."""
    im = terrain(1000, 800, seed=seed)
    boxes = [draw_person(im, 300, 300, G, jacket=(200, 25, 30)),
             draw_person(im, 700, 520, G, jacket=(240, 120, 10), lying=True)]
    ImageDraw.Draw(im).ellipse((820, 120, 850, 140), fill=(170, 170, 160))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)
    return [[round(v, 1) for v in b] for b in boxes]


@pytest.fixture()
def voc(tmp_path):
    root = tmp_path / "heridal_like"
    truth = {}
    for k in range(3):
        img = root / ("train" if k < 2 else "test") / f"img_{k}.jpg"
        boxes = _scene(img, k)
        truth[img.name] = boxes
        objs = "".join(f"<object><name>human</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax>"
                       f"<ymax>{b[3]}</ymax></bndbox></object>" for b in boxes)
        objs += "<object><name>rock</name><bndbox><xmin>820</xmin><ymin>120</ymin><xmax>850</xmax><ymax>140</ymax></bndbox></object>"
        img.with_suffix(".xml").write_text(f"<annotation><filename>{img.name}</filename>{objs}</annotation>")
    return root, truth


# ---------------------------------------------------------------- importer

def test_import_voc_with_class_map_and_splits(voc, tmp_path):
    root, _ = voc
    assert detect_format(root) == "voc"
    st = import_dataset(root, tmp_path / "ds", class_map={"human": "person"}, tile=320, negatives_per_image=1,
                        gsd_cm=1.64, log=lambda *a: None)
    assert st["images"] == 3 and st["dropped_classes"] == {"rock": 3}
    ds = Dataset(tmp_path / "ds")
    assert set(ds.classes) == {"person"}
    assert {s.split for s in ds.samples if "test" in s.source["photo"]} == {"val"}
    assert {s.split for s in ds.samples if "train" in s.source["photo"]} == {"train"}
    assert all(s.source["gsd_cm"] == 1.64 for s in ds.samples)
    negs = [s for s in ds.samples if not s.boxes]
    assert len(negs) == 3
    again = import_dataset(root, tmp_path / "ds", class_map={"human": "person"}, tile=320, log=lambda *a: None)
    assert again["positive"] == 0 and again["skipped_existing"] > 0


def test_import_yolo_and_coco(tmp_path):
    img = tmp_path / "y" / "images" / "train" / "a.jpg"
    _scene(img, 5)
    (tmp_path / "y" / "labels" / "train").mkdir(parents=True)
    (tmp_path / "y" / "labels" / "train" / "a.txt").write_text("0 0.3 0.375 0.03 0.03\n")
    (tmp_path / "y" / "data.yaml").write_text("names:\n  0: person\n")
    st = import_dataset(tmp_path / "y", tmp_path / "dy", fmt="yolo", tile=320, log=lambda *a: None)
    assert st["boxes"] == 1 and Dataset(tmp_path / "dy").classes == ["person"]
    c = tmp_path / "c"
    _scene(c / "b.jpg", 6)
    (c / "ann.json").write_text(json.dumps({"images": [{"id": 1, "file_name": "b.jpg", "width": 1000, "height": 800}],
                                            "annotations": [{"id": 1, "image_id": 1, "category_id": 3,
                                                             "bbox": [290, 290, 20, 20]}],
                                            "categories": [{"id": 3, "name": "person"}]}))
    st = import_dataset(c, tmp_path / "dc", tile=320, log=lambda *a: None)
    assert st["boxes"] == 1


# ---------------------------------------------------------------- scorecard

def test_matching_is_lenient_for_small_boxes():
    assert matches([10, 10, 20, 20], [12, 12, 22, 22])
    assert matches([0, 0, 40, 40], [15, 15, 20, 20])  # loose box around a tiny target
    assert not matches([0, 0, 10, 10], [50, 50, 60, 60])


def test_summary_maths():
    spec = SystemSpec("x", default_threshold=0.5)
    # 4 targets; predictions: TP .9, FP .8, TP .6, FP .3 ; per-GT best scores
    r = _summarise(spec, [(0.9, 1), (0.8, 0), (0.6, 1), (0.3, 0)], [0.9, 0.6, 0, 0], [20, 20, 20, 20], 4,
                   mp=20.9 * 2, secs=1.0)
    assert r["at_default"] == {"threshold": 0.5, "recall": 0.5, "precision": pytest.approx(2 / 3, abs=1e-3),
                               "fa_per_photo": 0.5}
    assert r["max_recall"] == 0.5
    assert r["recall_at_fa"]["0.5"] == 0.5 and r["recall_at_fa"]["1"] == 0.5
    assert 0 < r["ap"] <= 0.5


def test_evaluate_classic_end_to_end(voc, tmp_path):
    root, _ = voc
    import_dataset(root, tmp_path / "ds", class_map={"human": "person"}, tile=320, split_override="val", gsd_cm=1.64,
                   log=lambda *a: None)
    res = evaluate(tmp_path / "ds", [SystemSpec("colour", classic=True)], classes=["person"], log=lambda *a: None)
    r = res["systems"][0]
    assert res["targets"] == 6 and r["max_recall"] >= 0.8  # bright red / orange people are easy
    assert main(["eval", "-d", str(tmp_path / "ds"), "--classic", "-o", str(tmp_path / "sc")]) == 0
    assert (tmp_path / "sc" / "scorecard.html").exists()
    assert "window.DATA" in (tmp_path / "sc" / "scorecard.html").read_text()


# ---------------------------------------------------------------- verifier

def _const(path, out_name, arr, size):
    """ONNX model returning a fixed tensor per batch item (input still used)."""
    nodes = [helper.make_node("ReduceMean", ["images"], ["m"], axes=[1, 2, 3], keepdims=0),
             helper.make_node("Mul", ["m", "zero"], ["z"]),
             helper.make_node("Unsqueeze", ["z", "ax"], ["z2"]),
             helper.make_node("Add", ["c", "z2"], [out_name])]
    inits = [numpy_helper.from_array(np.array(0, np.float32), "zero"),
             numpy_helper.from_array(np.array([1], np.int64), "ax"),
             numpy_helper.from_array(arr.astype(np.float32), "c")]
    g = helper.make_graph(nodes, "v", [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["n", 3, size, size])],
                          [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", arr.shape[-1]])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def test_classifier_verifier(tmp_path):
    p = tmp_path / "cls.onnx"
    _const(p, "logits", np.array([0.0, 2.0]), 32)  # P(person) = sigmoid(2) ~ 0.88
    VerifierCard(type="classifier", input_size=32, classes=["background", "person"]).save(p)
    v = Verifier(p)
    probs = v.probability([Image.new("RGB", (10, 10))] * 3)
    assert probs == pytest.approx([1 / (1 + np.exp(-2))] * 3, abs=1e-4)


def test_embedding_verifier(tmp_path):
    p = tmp_path / "emb.onnx"
    _const(p, "embeddings", np.array([1.0, 0.0, 0.0]), 32)
    np.savez(p.with_suffix(".prompts.npz"), positive=np.array([[1.0, 0, 0]]), negative=np.array([[0, 1.0, 0]]))
    VerifierCard(type="embedding", input_size=32, logit_scale=10.0).save(p)
    v = Verifier(p)
    assert v.probability([Image.new("RGB", (8, 8))])[0] == pytest.approx(np.exp(10) / (np.exp(10) + 1), abs=1e-4)


def test_apply_verification_never_removes():
    hi = Detection(0, 0, 1, 1, 0.5, ["anomaly"], "x")
    lo = Detection(0, 0, 1, 1, 0.5, ["anomaly"], "y")
    apply_verification(hi, 0.9, "person")
    apply_verification(lo, 0.05, "person")
    assert hi.score > 0.5 and "verified" in hi.kinds
    assert 0 < lo.score < 0.5 and lo.score >= 0.5 * 0.5


def test_context_crop_centred_and_padded():
    im = Image.new("RGB", (100, 100), (255, 0, 0))
    c = context_crop(im, (0, 0, 10, 10), context=3.0, min_px=10)
    assert c.size == (30, 30)
    assert c.getpixel((0, 0)) == (0, 0, 0) and c.getpixel((15, 15)) == (255, 0, 0)  # padded beyond the edge


def test_train_crop_verifier(voc, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from isr.ml.verifier_build import train_crop_classifier
    root, _ = voc
    import_dataset(root, tmp_path / "ds", class_map={"human": "person"}, tile=320, negatives_per_image=2,
                   log=lambda *a: None)
    m = train_crop_classifier(tmp_path / "ds", tmp_path / "v.onnx", epochs=2, size=48, weights="none", batch=16,
                              log=lambda *a: None)
    assert m["classes"] == ["background", "person"]
    v = Verifier(tmp_path / "v.onnx")
    p = v.probability([Image.new("RGB", (20, 20)), Image.new("RGB", (20, 20), (200, 30, 30))])
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1))


# ---------------------------------------------------------------- quantisation

def test_quantize_keeps_detections(tmp_path):
    torch = pytest.importorskip("torch")
    from isr.ml.onnx_detector import ModelCard
    from isr.ml.quantize import compare, quantize_model

    class Tiny(torch.nn.Module):  # conv backbone + YOLO-layout head with a confident box at the centre
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.h = torch.nn.Conv2d(8, 6, 1)

        def forward(self, x):
            y = self.h(torch.relu(self.c(x)))[:, :, ::16, ::16].flatten(2)  # (1, 6, 16)
            base = torch.zeros_like(y)
            base[:, 0:4, 0] = torch.tensor([32.0, 32.0, 10.0, 10.0])
            base[:, 4, 0] = 0.9
            return base + 0 * y

    m = tmp_path / "tiny.onnx"
    torch.onnx.export(Tiny().eval(), torch.zeros(1, 3, 64, 64), str(m), input_names=["images"], output_names=["out"],
                      opset_version=17, dynamo=False)
    ModelCard(classes=["person", "dog"], input_size=64, tile=64).save(m)
    cal = tmp_path / "cal"
    for k in range(4):
        _scene(cal / f"{k}.jpg", k)
    q = quantize_model(m, tmp_path / "tiny.int8.onnx", cal, samples=4, log=lambda *a: None)
    assert q.exists() and q.with_suffix(".json").exists()
    r = compare(m, q, [cal / "0.jpg"])
    assert r["agreement"] == 1.0
