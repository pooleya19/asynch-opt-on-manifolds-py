import numpy as np
import multiprocessing

from RotationInvariantFormationControl import run_sim


def parameter_iterable_generator(fixed_parameter_dict, varying_parameter_dict):
    varying_parameter_names = list(varying_parameter_dict.keys())
    num_varying_parameters = len(varying_parameter_names)
    varying_parameter_sizes = np.zeros(num_varying_parameters)
    for i in range(num_varying_parameters):
        varying_parameter_sizes[i] = len(varying_parameter_dict[varying_parameter_names[i]])
    varying_parameter_sizes_cumprod = np.cumprod(varying_parameter_sizes)

    num_parameter_combinations = np.prod(varying_parameter_sizes)
    for i in range(num_parameter_combinations):
        output_parameter_dict = fixed_parameter_dict.copy()
        for j in range(num_varying_parameters):
            varying_parameter_name = varying_parameter_names[j]
            if i == 0:
            varying_parameter_index = 0
    



def main():
    results = None
    with multiprocessing.Pool() as pool:
        parameter_iterable = parameter_iterable_generator()
        results = pool.imap(run_sim, parameter_iterable)

    if results is None:
        raise RuntimeError("Pool terminated, but no results were returned.")
    else:
        num_results = len(results)
        print("Finished %g simulations.", num_results)
        for i in num_results:
            print("%g: awesome" % (i))


if __name__ == "__main__":
    main()