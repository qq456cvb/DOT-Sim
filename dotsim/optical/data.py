"""Data for the residual optical model.

``scripts/simulate_dataset.py`` writes, per recorded sequence, three videos: ``*_gt_colors.mp4`` (real tactile
images), ``*_depths.mp4`` (simulated depth x 3) and ``*_normals.mp4`` (simulated normals mapped to [0, 1]).
``RenderedSequenceDataset`` turns them into samples, ``scripts/build_webdataset.py`` stores those as WebDataset shards
and ``create_dataloader_from_webdataset`` reads the shards for training.
"""

from pathlib import Path

import imageio
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms


def read_mp4(path):
    return np.stack([frame for frame in imageio.get_reader(path)])


def to_pil(arr, is_diff=False):
    """C x H x W float array in [0, 1] (or [-1, 1] for differences) -> PIL image."""
    scaled = (arr + 1.0) / 2.0 if is_diff else arr
    scaled = np.clip(scaled, 0.0, 1.0)
    return Image.fromarray(np.moveaxis((scaled * 255.0).astype(np.uint8), 0, -1))


class RenderedSequenceDataset(torch.utils.data.Dataset):
    """One sample per frame of every ``*_gt_colors.mp4`` in ``fns``.

    Samples are dicts of PIL images: ``gt_rgb``, ``input_depth``, ``input_normal`` and, with ``use_diff``, the first
    frame of the sequence (``ref_rgb``, ``ref_depth``, ``ref_normal``), the other images then being differences to it.
    The simulated images are flipped vertically to match the camera.
    """

    def __init__(self, fns, use_diff=False):
        self.use_diff = use_diff
        self.fns = fns
        self.index_pairs = []
        for fn_idx, fn in enumerate(self.fns):
            n_frames = len(read_mp4(str(fn)))
            self.index_pairs += [(fn_idx, frame_idx) for frame_idx in range(n_frames)]
        self._cache = (None, None)

    def __len__(self):
        return len(self.index_pairs)

    def _load(self, fn_idx):
        if self._cache[0] != fn_idx:
            fn = str(self.fns[fn_idx])
            gt_colors = read_mp4(fn)
            rendered_depths = read_mp4(fn.replace("_gt_colors.mp4", "_depths.mp4"))
            rendered_normals = read_mp4(fn.replace("_gt_colors.mp4", "_normals.mp4"))
            # to N x C x H x W in [0, 1]
            gt_colors = (np.moveaxis(gt_colors, -1, 1) / 255.).astype(np.float32)
            rendered_depths = (np.moveaxis(rendered_depths[:, ::-1], -1, 1) / 255.).astype(np.float32)
            rendered_normals = (np.moveaxis(rendered_normals[:, ::-1], -1, 1) / 255.).astype(np.float32)
            self._cache = (fn_idx, (gt_colors, rendered_depths, rendered_normals))
        return self._cache[1]

    def __getitem__(self, idx):
        fn_idx, frame_idx = self.index_pairs[idx]
        gt_colors, rendered_depths, rendered_normals = self._load(fn_idx)
        gt_img, depth, normal = gt_colors[frame_idx], rendered_depths[frame_idx], rendered_normals[frame_idx]
        res = {"name": str(self.fns[fn_idx]), "frame_idx": frame_idx}
        if self.use_diff:
            ref_img, ref_depth, ref_normal = gt_colors[0], rendered_depths[0], rendered_normals[0]
            res.update({
                "input_depth": to_pil(depth - ref_depth, is_diff=True),
                "gt_rgb": to_pil(gt_img - ref_img, is_diff=True),
                "input_normal": to_pil(normal - ref_normal, is_diff=True),
                "ref_rgb": to_pil(ref_img),
                "ref_depth": to_pil(ref_depth),
                "ref_normal": to_pil(ref_normal),
            })
        else:
            res.update({
                "input_depth": to_pil(depth),
                "gt_rgb": to_pil(gt_img),
                "input_normal": to_pil(normal),
            })
        return res


def create_dataloader_from_webdataset(webdataset_dir, split, batch_size, num_workers=4, shuffle_buffer=1000, repeat=1,
                                      is_diff=True):
    """Loader over the ``<split>-*.tar`` shards in ``webdataset_dir``.

    Yields tuples (input_depth, gt_rgb, input_normal, ref_rgb, ref_depth, ref_normal) of batched tensors; with
    ``is_diff`` the first three are mapped back from [0, 1] to differences in [-1, 1].
    """
    to_tensor = transforms.ToTensor()
    webdataset_dir = Path(webdataset_dir)
    urls = sorted(webdataset_dir.glob(f"{split}-*.tar"))
    if not urls:
        raise FileNotFoundError(f"No shards matching '{split}-*.tar' in {webdataset_dir}")
    print(f"Found {len(urls)} shards for split '{split}'")

    # "file:" keeps Windows drive letters from being read as URL schemes
    dataset = wds.WebDataset([f"file:{u.resolve().as_posix()}" for u in urls])
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    if repeat > 1:
        dataset = dataset.repeat(repeat)
    dataset = dataset.decode("pil")

    def process_sample(sample):
        input_depth = to_tensor(sample["input_depth.png"])
        gt_rgb = to_tensor(sample["gt_rgb.png"])
        input_normal = to_tensor(sample["input_normal.png"])
        if is_diff:
            input_depth = input_depth * 2. - 1.
            gt_rgb = gt_rgb * 2. - 1.
            input_normal = input_normal * 2. - 1.
        refs = [to_tensor(sample[k]) if k in sample else None for k in ("ref_rgb.png", "ref_depth.png", "ref_normal.png")]
        return {
            "key": sample["__key__"],
            "input_depth": input_depth,
            "gt_rgb": gt_rgb,
            "input_normal": input_normal,
            "ref_rgb": refs[0],
            "ref_depth": refs[1],
            "ref_normal": refs[2],
            "metadata": sample["json"],
        }

    dataset = dataset.map(process_sample)
    dataset = dataset.to_tuple("input_depth", "gt_rgb", "input_normal", "ref_rgb", "ref_depth", "ref_normal")
    dataset = dataset.batched(batch_size)

    # Batching and shuffling happen in the WebDataset pipeline, so the loader itself does neither.
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, shuffle=False, pin_memory=True,
                               persistent_workers=num_workers > 0)
    if shuffle_buffer > 0:
        dataloader = dataloader.unbatched().shuffle(shuffle_buffer).batched(batch_size, partial=False)
    return dataloader
