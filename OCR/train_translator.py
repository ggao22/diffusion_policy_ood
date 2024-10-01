"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import hydra
from omegaconf import OmegaConf
import pathlib

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config'))
)
def main(main_cfg: OmegaConf):
    OmegaConf.resolve(main_cfg)
    


if __name__ == "__main__":
    main()
