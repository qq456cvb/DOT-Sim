"""Train the residual optical model (Sec. III-B) on the shards from build_webdataset.py.

    python scripts/train_optical.py data_root=/path/to/val_t1_t3_webdataset_pil

Each run gets its own folder under ``outputs/<date>/<time>/`` (Hydra); checkpoints are saved there every 10 epochs.
"""

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dotsim.optical import DeepLabModel


@hydra.main(version_base="1.1", config_path="../configs", config_name="optical")
def train(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    logger = False
    if cfg.use_wandb:
        logger = WandbLogger(project="DenseTact-image",
                             config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    model = DeepLabModel(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        callbacks=[ModelCheckpoint(save_last=True, every_n_epochs=10, save_top_k=-1)],
        limit_val_batches=1,
        check_val_every_n_epoch=5,
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=4,
    )
    trainer.fit(model)
    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
