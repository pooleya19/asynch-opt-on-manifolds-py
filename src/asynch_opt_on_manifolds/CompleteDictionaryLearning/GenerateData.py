import numpy as np
from pathlib import Path
from sklearn.datasets import load_digits, fetch_openml
from scipy.stats import ortho_group


def saveDataFloat32(Y, save_directory, file_name):
    file_name_new = "_32.".join(file_name.rsplit(".",1))
    file_path = str(save_directory.joinpath(file_name_new).resolve())
    print("(Saving 32-bit version... ", end='', flush=True)
    np.save(file_path, Y.astype(np.float32))
    print("Done!) ", end='', flush=True)


def saveSyntheticData(dim, num_samples, save_directory, file_name):
    print("Saving synthetic data... ", end='', flush=True)
    percent_nonzeros = 0.2
    std_dev = 0.2

    rng = np.random.default_rng(seed=0)
    A_true = ortho_group.rvs(dim=dim, random_state=rng)
    X_true = rng.normal(size=(dim,num_samples)) * (rng.uniform(size=(dim,num_samples)) < percent_nonzeros)
    Y = A_true @ X_true + rng.normal(loc=0, scale=std_dev, size=(dim,num_samples))
    # Y -= np.mean(Y, axis=1, keepdims=True)
    # sigma = 1/num_samples * Y@Y.T
    # Y = np.linalg.inv(scipy.linalg.sqrtm(sigma)) @ Y
    file_path = save_directory.joinpath(file_name)
    np.save(str(file_path.resolve()), Y)
    saveDataFloat32(Y, save_directory, file_name)
    print("Done!")


def saveDigitsData(k, save_directory, file_name):
    print("Saving digits... ", end='', flush=True)
    X, _ = load_digits(return_X_y=True)
    Y = X.T[:,0:k].copy()
    # Y -= np.mean(Y, axis=0, keepdims=True)
    # scale = np.linalg.norm(Y) / np.sqrt(Y.shape[1])
    # Y /= scale
    file_path = save_directory.joinpath(file_name)
    np.save(str(file_path.resolve()), Y)
    saveDataFloat32(Y, save_directory, file_name)
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
    saveDataFloat32(Y, save_directory, file_name)
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
    saveDataFloat32(Y, save_directory, file_name)
    print("Done!")


def main():
    save_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset")
    save_directory_path.mkdir(parents=True, exist_ok=True)
    saveSyntheticData(36, 10000, save_directory_path, "synthetic.npy")
    saveDigitsData(1500, save_directory_path, "digits.npy")
    # saveUSPSData(5000, save_directory_path, "usps.npy")
    # saveFashionMNISTData(10000, save_directory_path, "fashion_mnist.npy")


if __name__ == "__main__":
    main()