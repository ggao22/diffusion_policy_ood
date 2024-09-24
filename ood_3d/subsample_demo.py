import click
import numpy as np
import h5py
import shutil

@click.command()
@click.option('-i', '--input', required=True, type=str)
@click.option('-o', '--output', required=True, type=str)
@click.option('-s', '--sample_percent', required=True, type=float)
def main(input, output, sample_percent):
    shutil.copy(input, output)
    origin_file = h5py.File(input, 'r')
    n_demos = len(origin_file['data'])

    # modify action
    with h5py.File(output, 'r+') as out_file:
        for i in range(int(n_demos*sample_percent), n_demos):
            print(i)
            del out_file['data'][f'demo_{str(i)}']

if __name__ == "__main__":
    main()