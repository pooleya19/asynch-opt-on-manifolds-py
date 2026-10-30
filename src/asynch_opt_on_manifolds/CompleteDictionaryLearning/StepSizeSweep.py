import multiprocessing
from pathlib import Path
from threadpoolctl import threadpool_limits

from asynch_opt_on_manifolds.CompleteDictionaryLearning.Sim import run_sim


def run_sim_safely(parameters):
    try:
        threadpool_limits(1, "blas")
        return run_sim(parameters)
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def main():
    default_parameters = {
        "rng_seed" : 0,
        "N" : 8,
        "B_comp1" : 2,
        "B_comp2" : 10,
        "B_comm" : 500,
        "kmax" : 10000,
        "eta" : 0.01,
        "eta_synch" : 0.01,
        "beta" : 20,
        "sparsity_threshold" : 0.01,
        "m" : 5,
        "r" : 83,
        "data_file_name" : str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset").joinpath("fashion_mnist.npy").resolve()),
        "show_fig" : False,
        "save_final_fig_data" : False,
        "save_final_fig_data_name" : None,
        "save_video" : False,
        "save_video_name" : None,
        "save_data" : True,
        "save_data_name" : None,
    }

    values_rng_seed = [100,101,102]
    values_eta = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    values_B_N = [(500,64), (2000,8), (10,8)]
    save_data_path = Path(__file__).parents[3].joinpath("data").joinpath("dictionary_learning")

    with multiprocessing.Pool() as pool:
        parameter_sets = []

        for rng_seed in values_rng_seed:
            for eta in values_eta:
                for B,N in values_B_N:
                    parameter_set = default_parameters.copy()
                    parameter_set["rng_seed"] = rng_seed
                    parameter_set["eta"] = eta
                    parameter_set["eta_synch"] = eta
                    parameter_set["B_comm"] = B
                    parameter_set["N"] = N
                    parameter_set["save_data_name"] = str(save_data_path.joinpath("exp_step").joinpath("rng%g_Bcomm%g_N%g_eta%g.json" % (rng_seed, B, N, eta)).resolve())
                    parameter_sets.append(parameter_set)

        # Run all experiments
        num_sims = len(parameter_sets)
        print("Beginning %g simulations on %g workers:" % (num_sims, pool._processes))
        for sim_index, sim_time in enumerate(pool.imap_unordered(run_sim_safely, parameter_sets)):
            if sim_time is not None:
                print("Finished sim %g/%g in %.2f seconds." % (sim_index+1, num_sims, sim_time))
            else:
                print("Sim %g/%g failed." % (sim_index+1, num_sims))


if __name__ == "__main__":
    main()