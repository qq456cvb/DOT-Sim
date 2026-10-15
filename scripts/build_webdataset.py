"""Pack the simulated sequences into WebDataset shards for training the residual optical model.

The split files list ``*_gt_colors.mp4`` paths, one per line; the training list has the same name as the validation
list with "val" replaced by "train" (e.g. ``val_t1_t3.txt`` and ``train_t1_t3.txt``). Shards are written to
``<data_root>/<split name>_webdataset_pil/{train,val}-*.tar``.
"""

import argparse
from pathlib import Path

import webdataset as wds
from tqdm import tqdm

from dotsim.optical import RenderedSequenceDataset


def create_webdataset(dataset, output_dir, split_name, maxsize=1e9, maxcount=10000):
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"file:{(output_dir.resolve() / f'{split_name}-%06d.tar').as_posix()}"  # file: for Windows paths
    with wds.ShardWriter(pattern, maxsize=int(maxsize), maxcount=int(maxcount)) as sink:
        for index in tqdm(range(len(dataset)), desc=f"Writing {split_name} shards"):
            try:
                sample = dataset[index]
            except Exception as e:
                print(f"Skipping index {index} due to error: {e}")
                continue
            wds_sample = {
                "__key__": f"{split_name}_{Path(sample['name']).stem}_{sample['frame_idx']:04d}",
                "input_depth.png": sample["input_depth"],
                "gt_rgb.png": sample["gt_rgb"],
                "input_normal.png": sample["input_normal"],
                "json": {"name": sample["name"], "frame_idx": sample["frame_idx"]},
            }
            for key in ("ref_rgb", "ref_depth", "ref_normal"):
                if key in sample:
                    wds_sample[f"{key}.png"] = sample[key]
            sink.write(wds_sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_root", default="outputs/img_pred/data_E27575.0_nu0.303/es1t",
                        help="folder of simulated sequences (output of simulate_dataset.py)")
    parser.add_argument("--split_dir", default=None, help="folder with the split files (default: <data_root>/../split_fns)")
    parser.add_argument("--split_fn", default="val_t1_t3.txt", help="validation split file")
    parser.add_argument("--shard_maxsize", type=float, default=1e9, help="maximum shard size in bytes")
    parser.add_argument("--shard_maxcount", type=int, default=10000, help="maximum number of samples per shard")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_dir = Path(args.split_dir) if args.split_dir else data_root / "../split_fns"
    output_dir = data_root / args.split_fn.replace(".txt", "_webdataset_pil")
    val_fns = (split_dir / args.split_fn).read_text().splitlines()
    train_fns = (split_dir / args.split_fn.replace("val", "train")).read_text().splitlines()

    print("Initializing datasets...")
    train_dataset = RenderedSequenceDataset(train_fns, use_diff=True)
    val_dataset = RenderedSequenceDataset(val_fns, use_diff=True)
    print(f"Found {len(train_dataset)} training samples.")
    print(f"Found {len(val_dataset)} validation samples.")

    create_webdataset(train_dataset, output_dir, "train", maxsize=args.shard_maxsize, maxcount=args.shard_maxcount)
    create_webdataset(val_dataset, output_dir, "val", maxsize=args.shard_maxsize, maxcount=args.shard_maxcount)
    print(f"Shards written to {output_dir}")


if __name__ == "__main__":
    main()
