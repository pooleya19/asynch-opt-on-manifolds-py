import numpy as np
import multiprocessing
import os
from pathlib import Path

from asynch_opt_on_manifolds.RotationInvariantFormationControl.RotationInvariantFormationControl import run_sim


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
        parameter_set.append(output_parameter_dict)

    return parameter_set


def main():
    save_directory_path = str(Path(__file__).parents[3].joinpath("data").resolve())
    fixed_parameter_dict = {              "rng_seed":     0,
                                              "kmax":  10000,
                                               "lam":  0.01,
                                         "lam_synch":  0.01,
                                             "alpha":  0.01,
                            "graph_edge_probability":   0.8,
                                        "world_size":     1,
                                          "show_fig": False,
                                       "save_output":  True,
                               "save_directory_path": save_directory_path}

    with multiprocessing.Pool() as pool:
        # Test Experiment: N=10, B=[100,500,1000]
        # varying_parameter_dict = {"N":[10],
        #                           "B":[100,500,1000],
        #                           "save_file_name": ["exp1/sim_N<N>_B<B_comm>.json"]}
        # parameter_set = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict)

        # Experiment 1: N=10, B=[1,5,10,50,100,500,1000]
        varying_parameter_dict1 = {"rng_seed":[0,1,2,3,4],
                                   "N":[10],
                                   "B":[1,5,10,50,100,500,1000],
                                   "save_file_name": ["exp1/rng<rng_seed>_sim_N<N>_B<B_comm>.json"]}
        parameter_set1 = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict1)

        # Experiment 2: B=500, N=[5,10,20,50,100]
        varying_parameter_dict2 = {"rng_seed":[0,1,2,3,4],
                                   "N":[5,10,20,50,100],
                                   "B":[500],
                                   "save_file_name": ["exp2/rng<rng_seed>_sim_N<N>_B<B_comm>.json"]}
        parameter_set2 = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict2)

        # Experiment 3: N=10, B_comm=1000, B_comp=[1,5,10,50,100,500,1000]
        varying_parameter_dict3 = {"rng_seed":[0,1,2,3,4],
                                   "N":[10],
                                   "B_comm":[1000],
                                   "B_comp":[1,5,10,50,100,500,1000],
                                   "save_file_name": ["exp3/rng<rng_seed>_sim_N<N>_Bcomm<B_comm>_Bcomp<B_comp>.json"]}
        parameter_set3 = parameter_set_generator(fixed_parameter_dict, varying_parameter_dict3)

        # Run all experiments
        # parameter_sets = parameter_set
        # parameter_sets = parameter_set1 + parameter_set2 + parameter_set3
        parameter_sets = parameter_set1
        num_sims = len(parameter_sets)
        print("Beginning %g simulations:" % num_sims)
        for sim_index, sim_time in enumerate(pool.imap_unordered(run_sim, parameter_sets)):
            print("Finished sim %g/%g in %.2f seconds." % (sim_index+1, num_sims, sim_time))

if __name__ == "__main__":
    main()