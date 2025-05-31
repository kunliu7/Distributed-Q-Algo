import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    n_files = 150

    args = parser.parse_args()
    fids = []
    filenames = os.listdir()
    filenames = [f for f in filenames if '.txt' in f]
    if len(filenames) < 150:
        print('Must have >= 150 files')

    for filename in filenames:
        path = f'{args.output_folder}/{filename}'
        with open(path) as file:
            fids.append(float(file.read()))

    mean = np.mean(fids)
    std = np.std(fids)
    print(mean, std)

if __name__ == "__main__":
    main()