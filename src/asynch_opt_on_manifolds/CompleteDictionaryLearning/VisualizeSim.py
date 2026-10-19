import numpy as np
import os
from pathlib import Path

from asynch_opt_on_manifolds.DictionaryLearning.DictionaryLearning import run_sim


def main():
    save_directory_path = str(Path(__file__).parents[3].joinpath("figures").joinpath("DictionaryLearning").resolve())
    parameters = {
        "rng_seed": 0,
        "N": 10,
        "n": 300,
        "B": 500,
        # "B_comm": 100,
        # "B_comp": 100,
        "kmax": 20000,
        "gamma": 10,
        "gamma_synch": 10,
        "lam": 0.1,
        "eps": 1e-7,
        "sparsity_threshold": 0.005,
        "data_directory_path": str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset").resolve()),
        "data_file_name": "usps.npy",
        "show_fig": True,
        "save_fig": True,
        "save_output": False,
        "save_directory_path": save_directory_path,
        "save_file_name": None,
        "save_fig_name": "visualization.svg"
    }
    run_sim(parameters)

if __name__ == "__main__":
    main()