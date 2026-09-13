import numpy as np
from pathlib import Path


def generateSyntheticData(k, n, d):
    nonzeros_per_column = 0.2*n
    std_dev = 0.2

    rng = np.random.default_rng(seed=0)
    D_true = rng.normal(size=(d,n))
    D_true = D_true / np.linalg.norm(D_true, axis=0)
    C_true = rng.normal(size=(n,k)) * (rng.uniform(size=(n,k)) < nonzeros_per_column/n)
    Y = D_true @ C_true + rng.normal(loc=0, scale=std_dev, size=(d,k))
    return Y


def main():
    save_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset")
    dataset_file_name = "synthetic.npy"
    num_images = 500
    num_atoms = 30
    image_dimension = 25

    print("Generating data... ", end='', flush=True)
    data = generateSyntheticData(num_images, num_atoms, image_dimension)
    print("Done!")

    save_directory_path.mkdir(parents=True, exist_ok=True)
    save_file_path = str(save_directory_path.joinpath(dataset_file_name).resolve())
    print("Saving data to %s... "%save_file_path, end='', flush=True)
    np.save(save_file_path, data)
    print("Done!")


if __name__ == "__main__":
    main()