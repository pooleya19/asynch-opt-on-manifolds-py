import numpy as np
from pathlib import Path
from sklearn.datasets import load_digits, fetch_openml


def saveSyntheticData(k, n, d, save_directory, file_name):
    print("Saving synthetic data... ", end='', flush=True)
    nonzeros_per_column = 0.2*n
    std_dev = 0.2

    rng = np.random.default_rng(seed=0)
    D_true = rng.normal(size=(d,n))
    D_true = D_true / np.linalg.norm(D_true, axis=0)
    C_true = rng.normal(size=(n,k)) * (rng.uniform(size=(n,k)) < nonzeros_per_column/n)
    Y = D_true @ C_true + rng.normal(loc=0, scale=std_dev, size=(d,k))
    Y -= np.mean(Y, axis=0, keepdims=True)
    scale = np.linalg.norm(Y) / np.sqrt(Y.shape[1])
    Y /= scale
    file_path = save_directory.joinpath(file_name)
    np.save(str(file_path.resolve()), Y)
    print("Done!")


def saveDigitsData(k, save_directory, file_name):
    print("Saving digits... ", end='', flush=True)
    X, _ = load_digits(return_X_y=True)
    Y = X.T[:,0:k].copy()
    Y -= np.mean(Y, axis=0, keepdims=True)
    scale = np.linalg.norm(Y) / np.sqrt(Y.shape[1])
    Y /= scale
    file_path = save_directory.joinpath(file_name)
    np.save(str(file_path.resolve()), Y)
    print("Done!")


def saveUSPSData(k, save_directory, file_name):
    print("Saving USPS... ", end='', flush=True)
    X, _ = fetch_openml('USPS', version=2, return_X_y=True, as_frame=False, parser="liac-arff")
    Y = X.T[:,0:k].copy()
    Y -= np.mean(Y, axis=0, keepdims=True)
    scale = np.linalg.norm(Y) / np.sqrt(Y.shape[1])
    Y /= scale
    file_path = save_directory.joinpath(file_name)
    np.save(str(file_path.resolve()), Y)
    print("Done!")


def saveFashionMNISTData(k, save_directory, file_name):
    print("Saving Fashion-MNIST... ", end='', flush=True)
    X, _ = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    Y = X.T[:,0:k].copy()
    Y -= np.mean(Y, axis=0, keepdims=True)
    scale = np.linalg.norm(Y) / np.sqrt(Y.shape[1])
    Y /= scale
    file_path = save_directory.joinpath(file_name)
    np.save(str(file_path.resolve()), Y)
    print("Done!")


def main():
    save_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset")
    save_directory_path.mkdir(parents=True, exist_ok=True)
    saveSyntheticData(500, 30, 25, save_directory_path, "synthetic.npy")
    saveDigitsData(1500, save_directory_path, "digits.npy")
    saveUSPSData(5000, save_directory_path, "usps.npy")
    saveFashionMNISTData(10000, save_directory_path, "fashion_mnist.npy")


if __name__ == "__main__":
    main()