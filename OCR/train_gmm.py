"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config'))
)
def main(main_cfg: OmegaConf):
    print(main_cfg)
    OmegaConf.resolve(main_cfg)

    translator_yaml = pathlib.Path(__file__).parent.parent.joinpath(
        'diffusion_policy','config') / 'train_diffusion_unet_lowdim_obsact_workspace.yaml'
    translator_config = OmegaConf.load(translator_yaml)
    print(main_cfg)
    


if __name__ == "__main__":
    main()
