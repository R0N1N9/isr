import json

import numpy as np
import pytest
from PIL import Image

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from isr.cli import main  # noqa: E402
from isr.ml.dataset import Dataset, add_from_triage, export  # noqa: E402
from isr.ml.onnx_detector import ModelCard, OnnxDetector, nms  # noqa: E402

S = 64  # model input size for the synthetic models


def _const_model(path, outputs: dict[str, np.ndarray], size=S, extra_input=False):
    """An ONNX model whose outputs are fixed tensors (plus 0 x mean(input), so the input is used)."""
    nodes = [helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
             helper.make_node("Mul", ["m", "zero"], ["z"])]
    inits = [numpy_helper.from_array(np.array(0, np.float32), "zero")]
    outs = []
    for name, arr in outputs.items():
        inits.append(numpy_helper.from_array(arr.astype(np.float32), f"c_{name}"))
        if arr.dtype.kind in "iu":
            nodes.append(helper.make_node("Identity", [f"c_{name}"], [f"{name}_f"]))
            nodes.append(helper.make_node("Cast", [f"{name}_f"], [name], to=TensorProto.INT64))
            outs.append(helper.make_tensor_value_info(name, TensorProto.INT64, list(arr.shape)))
        else:
            nodes.append(helper.make_node("Add", [f"c_{name}", "z"], [name]))
            outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape)))
    ins = [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])]
    if extra_input:
        ins.append(helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2]))
    g = helper.make_graph(nodes, "const", ins, outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


def _yolo(path):
    # 3 anchors: two overlapping person boxes (NMS keeps one) and a dog box
    o = np.zeros((1, 6, 3), np.float32)
    o[0, :, 0] = [32, 32, 10, 10, 0.9, 0.0]
    o[0, :, 1] = [33, 32, 10, 10, 0.8, 0.0]
    o[0, :, 2] = [10, 10, 6, 6, 0.0, 0.6]
    _const_model(path, {"output0": o})
    ModelCard(classes=["person", "dog"]).save(path)
    return path


def test_nms():
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert sorted(nms(b, np.array([0.9, 0.8, 0.7]))) == [0, 2]


def test_yolo_layout(tmp_path):
    det = OnnxDetector(_yolo(tmp_path / "y.onnx"), conf=0.25, tile=S)
    assert det.format == "yolo" and det.class_names == ["person", "dog"]
    out = det.detect_image(Image.new("RGB", (S, S)))
    assert [(round(d[0]), round(d[1]), d[5]) for d in out] == [(27, 27, "person"), (7, 7, "dog")]


def test_tiles_map_back_to_photo_coordinates(tmp_path):
    det = OnnxDetector(_yolo(tmp_path / "y.onnx"), conf=0.5, tile=S, overlap=0.0)
    out = det.detect_image(Image.new("RGB", (2 * S, S)))  # two tiles side by side
    xs = sorted(round(d[0]) for d in out if d[5] == "person")
    assert xs == [27, 27 + S]


def test_class_filter(tmp_path):
    det = OnnxDetector(_yolo(tmp_path / "y.onnx"), conf=0.25, tile=S, classes={"dog"})
    assert [d[5] for d in det.detect_image(Image.new("RGB", (S, S)))] == ["dog"]


def test_detr_layout(tmp_path):
    boxes = np.array([[[0.5, 0.5, 0.25, 0.25], [0.1, 0.1, 0.1, 0.1]]], np.float32)
    logits = np.array([[[4.0, -4.0], [-4.0, -4.0]]], np.float32)  # query 0 = class 0, query 1 = nothing
    p = _const_model(tmp_path / "d.onnx", {"dets": boxes, "labels": logits})
    det = OnnxDetector(p, conf=0.5, tile=S)
    assert det.format == "detr" and det.normalize == "imagenet"
    out = det.detect_image(Image.new("RGB", (S, S)))
    assert len(out) == 1 and [round(v) for v in out[0][:4]] == [24, 24, 40, 40] and out[0][5] == "class_0"


def test_yolo_e2e_layout(tmp_path):
    o = np.array([[[10, 10, 20, 30, 0.9, 1], [0, 0, 1, 1, 0.1, 0]]], np.float32)
    det = OnnxDetector(_const_model(tmp_path / "e.onnx", {"output0": o}), conf=0.5, tile=S)
    assert det.format == "yolo_e2e"
    out = det.detect_image(Image.new("RGB", (S, S)))
    assert [round(v) for v in out[0][:4]] == [10, 10, 20, 30] and out[0][5] == "class_1"


def test_dfine_layout(tmp_path):
    p = _const_model(tmp_path / "f.onnx", {"labels": np.array([[2, 0]], np.int64),
                                           "boxes": np.array([[[5, 6, 25, 26], [0, 0, 2, 2]]], np.float32),
                                           "scores": np.array([[0.8, 0.1]], np.float32)}, extra_input=True)
    det = OnnxDetector(p, conf=0.5, tile=S)
    assert det.format == "dfine"
    out = det.detect_image(Image.new("RGB", (S, S)))
    assert len(out) == 1 and [round(v) for v in out[0][:4]] == [5, 6, 25, 26] and out[0][5] == "class_2"


def test_model_cli(tmp_path, capsys):
    p = _yolo(tmp_path / "y.onnx")
    Image.new("RGB", (S, S)).save(tmp_path / "p.jpg")
    assert main(["model", "card", str(p), "--tile", "64", "--notes", "test"]) == 0
    assert json.loads((tmp_path / "y.json").read_text())["tile"] == 64
    assert main(["model", "info", str(p), "--photo", str(tmp_path / "p.jpg")]) == 0
    assert "person" in capsys.readouterr().out


# ---------------------------------------------------------------- dataset

def _run(tmp_path, generated="2026-11-19 10:00:00"):
    rng = np.random.default_rng(0)
    photos = []
    for k in range(2):
        f = tmp_path / "card" / f"DJI_{k}.JPG"
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((rng.random((800, 1000, 3)) * 255).astype(np.uint8)).save(f)
        photos.append({"file": str(f), "rel": f.name, "detections": [
            {"id": f"DJI_{k}-1", "box": [400, 300, 440, 330], "lat": 1, "lon": 2},
            {"id": f"DJI_{k}-2", "box": [100, 600, 130, 640], "lat": 1, "lon": 2},
            {"id": f"DJI_{k}-3", "box": [900, 100, 920, 120], "lat": 1, "lon": 2}]})
    out = tmp_path / "triage"
    out.mkdir()
    (out / "results.json").write_text(json.dumps({"summary": {"generated": generated, "source": "card"},
                                                  "photos": photos}))
    labels = {"generated": generated, "labels": {
        "DJI_0-1": {"status": "confirmed", "class": "person"}, "DJI_0-2": {"status": "dismissed"},
        "DJI_1-1": {"status": "confirmed", "class": "vehicle"}, "DJI_1-3": {"status": "dismissed"}}}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    return out, tmp_path / "labels.json"


def test_dataset_from_reviews(tmp_path):
    tri, lab = _run(tmp_path)
    res = add_from_triage(tri, lab, tmp_path / "ds", tile=320, pad=0.0, log=lambda *a: None)
    assert res["added"]["positive"] == 2 and res["added"]["negative"] == 2
    ds = Dataset(tmp_path / "ds")
    for s in ds.samples:
        assert (tmp_path / "ds" / s.file).exists()
        assert {x.source["photo"] for x in ds.samples if x.split == s.split} >= {s.source["photo"]}
        if s.kind == "positive":
            x0, y0, x1, y1 = s.boxes[0]["box"]
            tx, ty = s.source["tile_xy"]
            assert [x0 + tx, y0 + ty, x1 + tx, y1 + ty] == pytest.approx([400, 300, 440, 330], abs=0.2)
        else:
            assert s.boxes == []
    # each photo goes wholly to one split
    by_photo = {}
    for s in ds.samples:
        by_photo.setdefault(s.source["photo"], set()).add(s.split)
    assert all(len(v) == 1 for v in by_photo.values())
    # re-adding the same review adds nothing
    again = add_from_triage(tri, lab, tmp_path / "ds", tile=320, log=lambda *a: None)
    assert again["added"]["skipped_existing"] == 4 and len(Dataset(tmp_path / "ds").samples) == 4


def test_labels_from_other_run_rejected(tmp_path):
    tri, lab = _run(tmp_path)
    d = json.loads(lab.read_text())
    d["generated"] = "2020-01-01 00:00:00"
    lab.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        add_from_triage(tri, lab, tmp_path / "ds", log=lambda *a: None)


def test_exports(tmp_path):
    tri, lab = _run(tmp_path)
    add_from_triage(tri, lab, tmp_path / "ds", tile=320, pad=0.0, log=lambda *a: None)
    cfg = export(tmp_path / "ds", tmp_path / "yolo", "yolo", ["person", "vehicle"])
    assert "names:" in cfg.read_text() and "1: vehicle" in cfg.read_text()
    lines = [ln for f in (tmp_path / "yolo" / "labels").rglob("*.txt") for ln in f.read_text().splitlines()]
    assert sorted(ln.split()[0] for ln in lines) == ["0", "1"]
    for ln in lines:
        c, cx, cy, w, h = ln.split()
        assert 0 < float(cx) < 1 and 0 < float(w) <= 1 and float(w) == pytest.approx(40 / 320, abs=1e-4)
    export(tmp_path / "ds", tmp_path / "coco", "coco")
    tr = json.loads((tmp_path / "coco" / "train" / "_annotations.coco.json").read_text())
    va = json.loads((tmp_path / "coco" / "valid" / "_annotations.coco.json").read_text())
    assert len(tr["images"]) + len(va["images"]) == 4
    assert {c["name"] for c in tr["categories"]} == {"person", "vehicle"}
    for a in tr["annotations"] + va["annotations"]:
        assert a["bbox"][2] == pytest.approx(40, abs=0.2) and a["category_id"] >= 1


def test_dataset_cli(tmp_path, capsys):
    tri, lab = _run(tmp_path)
    assert main(["dataset", "add", str(tri), "--labels", str(lab), "-d", str(tmp_path / "ds"), "--tile", "320"]) == 0
    assert main(["dataset", "stats", "-d", str(tmp_path / "ds")]) == 0
    assert main(["dataset", "export", "-d", str(tmp_path / "ds"), "--format", "coco", "-o", str(tmp_path / "c")]) == 0
    assert "positive_tiles" in capsys.readouterr().out
