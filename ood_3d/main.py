import click

from trainer import train_recovery_policy
from tester import test_rand
from config import cfg


@click.command()
@click.option('-m', '--mode', required=True, type=str)
def main(mode):
    if mode=='train':
        train_recovery_policy(cfg)
    elif mode=='test':
        test_rand(cfg)
    else:
        raise Exception() 



if __name__ == "__main__":
    main()