import numpy as np
import multiprocessing

from asynch_opt_on_manifolds.RotationInvariantFormationControl.RotationInvariantFormationControl import run_sim


def parameter_iterable_generator(fixed_parameter_dict, varying_parameter_dict):
    varying_parameter_names = list(varying_parameter_dict.keys())
    num_varying_parameters = len(varying_parameter_names)
    varying_parameter_sizes = np.zeros(num_varying_parameters)
    for i in range(num_varying_parameters):
        varying_parameter_sizes[i] = len(varying_parameter_dict[varying_parameter_names[i]])
    varying_parameter_sizes_cumprod = np.cumprod(varying_parameter_sizes)

    num_parameter_combinations = int(np.prod(varying_parameter_sizes))
    for i in range(num_parameter_combinations):
        output_parameter_dict = fixed_parameter_dict.copy()
        output_parameter_dict["sim_index"] = i
        output_parameter_dict["num_sims"] = num_parameter_combinations
        for j in range(num_varying_parameters):
            varying_parameter_name = varying_parameter_names[j]
            if j == 0:
                varying_parameter_index = i % varying_parameter_sizes[0]
            else:
                varying_parameter_index = (i // varying_parameter_sizes_cumprod[j-1]) % varying_parameter_sizes[j]
            output_parameter_dict[varying_parameter_name] = varying_parameter_dict[varying_parameter_name][int(varying_parameter_index)]
        yield output_parameter_dict


def main():
    # Experiment 1: N=10, B=[1,5,10,50,100,500,1000]
    # Experiment 2: B=500, N=[5,10,20,50,100]
    # Experiment 3: N=10, B_comm=1000, B_comp=[1,5,10,50,100,500,1000]
    fixed_parameter_dict = {              "rng_seed":     0,
                                              "kmax":  1000,
                                               "lam":  0.01,
                                         "lam_synch":  0.01,
                                             "alpha":  0.01,
                            "graph_edge_probability":   0.8,
                                        "world_size":     1,
                                          "show_fig": False}
    # varying_parameter_dict = {"N":[5,10,20,50,100],
    #                           "B":[1,5,10,50,100,500]}
    varying_parameter_dict = {"N":[5,10],
                              "B":[100,500]}

    parameter_iterable = parameter_iterable_generator(fixed_parameter_dict, varying_parameter_dict)

    results = None
    with multiprocessing.Pool() as pool:
        results = pool.imap(run_sim, parameter_iterable)

    if results is None:
        raise RuntimeError("Pool terminated, but no results were returned.")
    else:
        for result in results:
            print("awesome")


if __name__ == "__main__":
    main()