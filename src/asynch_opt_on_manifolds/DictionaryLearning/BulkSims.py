import numpy as np
import multiprocessing
from pathlib import Path
from signal import signal, SIGINT
from threadpoolctl import threadpool_limits

from asynch_opt_on_manifolds.DictionaryLearning.DictionaryLearning import run_sim


def initializer():
    signal(SIGINT, lambda: None)
    threadpool_limits(1, "blas")


def parameter_set_generator(fixed_parameter_dict, varying_parameter_dict):
    parameter_set = []

    varying_parameter_names = list(varying_parameter_dict.keys())
    num_varying_parameters = len(varying_parameter_names)
    varying_parameter_sizes = np.zeros(num_varying_parameters)
    for i in range(num_varying_parameters):
        varying_parameter_sizes[i] = len(varying_parameter_dict[varying_parameter_names[i]])
    varying_parameter_sizes_cumprod = np.cumprod(varying_parameter_sizes)

    num_parameter_combinations = int(np.prod(varying_parameter_sizes))
    for i in range(num_parameter_combinations):
        output_parameter_dict = fixed_parameter_dict.copy()
        for j in range(num_varying_parameters):
            varying_parameter_name = varying_parameter_names[j]
            if j == 0:
                varying_parameter_index = i % varying_parameter_sizes[0]
            else:
                varying_parameter_index = (i // varying_parameter_sizes_cumprod[j-1]) % varying_parameter_sizes[j]
            output_parameter_dict[varying_parameter_name] = varying_parameter_dict[varying_parameter_name][int(varying_parameter_index)]

        # Process B_comm and B_calc
        if "B_comm_calc_ratio" in output_parameter_dict:
            output_parameter_dict["B_calc"] = max(1, output_parameter_dict["B_comm"]//output_parameter_dict["B_comm_calc_ratio"])
            del output_parameter_dict["B_comm_calc_ratio"]
        
        parameter_set.append(output_parameter_dict)

    return parameter_set


def main():
    save_directory_path = str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_results").resolve())
    fixed_parameter_dict = {
        "n": 1000,
        "kmax": 10000,
        "gamma": 5,
        "gamma_synch": 20,
        "lam": 0.01,
        "eps": 1e-7,
        "sparsity_threshold": 0.005,
        "data_directory_path": str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset").resolve()),
        "data_file_name": "fashion_mnist_32.npy",
        "show_fig": False,
        "save_fig": False,
        "save_output": True,
        "save_directory_path": save_directory_path,
        "save_fig_name": None
    }

    with multiprocessing.Pool(initializer=initializer) as pool:
        # varying_parameter_dict0 = {
        #     "rng_seed" : [0],
        #     "N" : [100],
        #     "B_comm" : [100],
        #     "B_comm_calc_ratio" : [1],
        #     "save_file_name": ["test/exp.json"],
        # }
        # parameter_set0 = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict0)

        # Experiment 1: Vary B (B_comm = B_calc)
        varying_parameter_dict1 = {
            "rng_seed":[0,1,2,3,4],
            "N":[10],
            "B_comm":[10,50,100,500,1000,2000],
            "B_comm_calc_ratio":[1],
            "save_file_name":["exp1/rng<rng_seed>_N<N>_B<B_comm>.json"]
        }
        parameter_set1 = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict1)

        # Experiment 2: Vary N
        varying_parameter_dict2 = {
            "rng_seed":[0,1,2,3,4],
            "N":[5,10,20,50,100],
            "B_comm":[500],
            "B_calc":[500],
            "save_file_name":["exp2/rng<rng_seed>_N<N>_B<B_comm>.json"]
        }
        parameter_set2 = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict2)

        # Experiment 3: Vary B_comm with fixed B_calc
        varying_parameter_dict3 = {
            "rng_seed":[0,1,2,3,4],
            "N":[10],
            "B_comm":[10,50,100,500,1000,2000],
            "B_calc":[10],
            "save_file_name":["exp3/rng<rng_seed>_N<N>_B<B_comm>.json"]
        }
        parameter_set3 = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict3)

        # Run all experiments
        parameter_sets = []
        # parameter_sets += parameter_set0
        parameter_sets += parameter_set1
        parameter_sets += parameter_set2
        parameter_sets += parameter_set3
        num_sims = len(parameter_sets)
        print("Beginning %g simulations on %g workers:" % (num_sims, pool._processes))
        for sim_index, sim_time in enumerate(pool.imap_unordered(run_sim, parameter_sets)):
            print("Finished sim %g/%g in %.2f seconds." % (sim_index+1, num_sims, sim_time))

if __name__ == "__main__":
    main()