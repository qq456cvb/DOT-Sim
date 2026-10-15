"""Smoke test of the optical pipeline on tiny synthetic videos: dataset -> shards -> training step -> evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_webdataset  # noqa: E402
import eval_optical  # noqa: E402
import simulate_dataset  # noqa: E402
from dotsim.optical import DeepLabModel, RenderedSequenceDataset  # noqa: E402

H, W, FRAMES = 48, 64, 4


def write_sequence(stem, rng):
    """Videos in the format written by simulate_dataset.py."""
    depths = [rng.uniform(0, 0.3, (H, W, 1)) for _ in range(FRAMES)]
    normals = [rng.uniform(-1, 1, (H, W, 3)) for _ in range(FRAMES)]
    colors = [rng.uniform(0, 1, (H, W, 3)) for _ in range(FRAMES)]
    simulate_dataset.write_video(f"{stem}_normals.mp4", [(np.clip((n + 1.) / 2., 0, 1) * 255).astype(np.uint8) for n in normals], 2.4)
    simulate_dataset.write_video(f"{stem}_depths.mp4", [(np.clip(d * 3, 0, 1) * 255).astype(np.uint8) for d in depths], 2.4)
    simulate_dataset.write_video(f"{stem}_gt_colors.mp4", [(np.clip(c, 0, 1) * 255).astype(np.uint8) for c in colors], 2.4)
    return f"{stem}_gt_colors.mp4"


def test_optical_pipeline(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    fns = [write_sequence(str(tmp_path / f"t1_seq{k}"), rng) for k in range(2)]

    dataset = RenderedSequenceDataset(fns, use_diff=True)
    assert len(dataset) == 2 * FRAMES
    sample = dataset[1]
    assert sample["gt_rgb"].size == (W, H) and sample["input_depth"].size == (W, H)

    shards = tmp_path / "shards"
    build_webdataset.create_webdataset(dataset, shards, "train")
    build_webdataset.create_webdataset(dataset, shards, "val")
    assert sorted(p.name for p in shards.glob("*.tar")) == ["train-000000.tar", "val-000000.tar"]

    cfg = OmegaConf.create({"data_root": str(shards), "use_diff": True, "use_gt": False, "lr": 3e-4,
                            "weight_decay": 1e-4, "batch_size": 2, "num_workers": 0, "shuffle_buffer": 4,
                            "val_num_workers": 0, "pretrained_backbone": False})
    model = DeepLabModel(cfg)
    trainer = pl.Trainer(accelerator="cpu", fast_dev_run=True, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)

    split = tmp_path / "val.txt"
    split.write_text("\n".join(fns))
    cfg_file = tmp_path / "config.yaml"
    OmegaConf.save(cfg, cfg_file)
    for extra in ([], ["--rescale"]):
        monkeypatch.setattr(sys, "argv", ["eval_optical.py", "--checkpoint", str(ckpt), "--config", str(cfg_file),
                                          "--split_fn", str(split), "--device", "cpu", *extra])
        eval_optical.main()
        out = capsys.readouterr().out
        psnr = float(out.split("Val PSNR:")[1].split()[0])
        assert np.isfinite(psnr)
