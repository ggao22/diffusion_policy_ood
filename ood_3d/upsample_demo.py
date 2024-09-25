import click
import numpy as np
import h5py
import shutil

@click.command()
@click.option('-i', '--input', required=True, type=str)
@click.option('-o', '--output', required=True, type=str)
def main(input, output):
    shutil.copy(input, output)

    # modify action
    with h5py.File(input, 'r') as in_file:
        with h5py.File(output, 'a') as out_file:
            for i in range(len(in_file['data'])):
                item = in_file['data'][f'demo_{str(i)}']
                in_file['data'].copy(item, out_file['data'], f"demo_{str(i+len(in_file['data']))}")

if __name__ == "__main__":
    main()