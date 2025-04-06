import json
import os
from typing import Any, Dict, Tuple
import argparse
import pytorch_lightning as pl
import torch
import hydra

torch.set_float32_matmul_precision("high")

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def print_only(message: str):
    """Prints a message only on rank 0."""
    print(message)
    
def train(cfg: DictConfig, args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Validate checkpoint path
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")
    
    # instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="test")  # Explicitly call setup
    
    # instantiate models
    print_only(f"Instantiating decouple model <{cfg.decouple_model._target_}>")
    decouple_model: torch.nn.Module = hydra.utils.instantiate(cfg.decouple_model)
    
    try:
        state_dict = torch.load(cfg.speechtokenizer_path, map_location="cpu")
        decouple_model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load decouple model weights: {str(e)}")
    
    print_only(f"Instantiating detect model <{cfg.detect_model._target_}>")
    detect_model: torch.nn.Module = hydra.utils.instantiate(cfg.detect_model)
    
    # instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        decouple_model=decouple_model,
        detect_model=detect_model,
    )
    
    # Configure trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=False,  # Disable logging during test if not needed
        enable_checkpointing=False,  # Disable checkpointing during test
    )
    
    # Run test
    try:
        results = trainer.test(
            system,
            datamodule=datamodule,
            ckpt_path=args.ckpt_path,
            verbose=True
        )
        return results
    except Exception as e:
        raise RuntimeError(f"Testing failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="local/conf.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to checkpoint file for testing",
    )
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.conf_dir):
        raise FileNotFoundError(f"Config file not found: {args.conf_dir}")
    
    cfg = OmegaConf.load(args.conf_dir)
    
    # Create output directory
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    # Run training/test
    try:
        train(cfg, args)
    except Exception as e:
        print_only(f"Error occurred: {str(e)}")
        raise
