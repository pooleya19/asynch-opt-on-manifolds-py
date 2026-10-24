import multiprocessing
from pathlib import Path

from asynch_opt_on_manifolds.RobustFormationControl.Sim import run_sim


def run_sim_safely(parameters):
    try:
        return run_sim(parameters)
    except:
        return None


def main():
    default_parameters = {
        "rng_seed" : 0,
        "N" : 12,
        "B_comp1" : 2,
        "B_comp2" : 10,
        "B_comm" : 10,
        "kmax" : 5000,
        "eta" : 0.2,
        "eta_synch" : 0.2,
        "alpha" : 1e-5,
        "delta" : 1,
        "show_fig" : False,
        "save_final_fig" : False,
        "save_final_fig_name" : None,
        "save_video" : False,
        "save_video_name" : None,
        "save_data" : True,
        "save_data_name" : None,
    }

    values_rng_seed = [0,1,2,3,4]
    values_B_comm = [10,20,50,100,200,500,1000,2000]
    values_N = [4,8,16,32,64,128,256]
    save_data_path = Path(__file__).parents[3].joinpath("data").joinpath("formation_control")
    save_figure_path = Path(__file__).parents[3].joinpath("figures").joinpath("formation_control")

    with multiprocessing.Pool() as pool:
        parameter_sets = []

        # Experiment 1: Vary B_comm
        for rng_seed in values_rng_seed:
            for B_comm in values_B_comm:
                parameter_set = default_parameters.copy()
                parameter_set["rng_seed"] = rng_seed
                parameter_set["B_comm"] = B_comm
                parameter_set["save_data_name"] = str(save_data_path.joinpath("exp1").joinpath("rng%g_Bcomm%g.json" % (rng_seed, B_comm)).resolve())
                parameter_sets.append(parameter_set)

        # Experiment 2: Vary N
        for rng_seed in values_rng_seed:
            for N in values_N:
                parameter_set = default_parameters.copy()
                parameter_set["rng_seed"] = rng_seed
                parameter_set["N"] = N
                parameter_set["save_data_name"] = str(save_data_path.joinpath("exp2").joinpath("rng%g_N%g.json" % (rng_seed, N)).resolve())
                parameter_sets.append(parameter_set)

        # Experiment 3: Render figure and video
        parameter_set = default_parameters.copy()
        parameter_set["save_final_fig"] = True
        parameter_set["save_final_fig_name"] = str(save_figure_path.joinpath("figure.pdf").resolve())
        parameter_set["save_video"] = True
        parameter_set["save_video_name"] = str(save_figure_path.joinpath("video.mp4").resolve())
        parameter_set["save_data"] = False
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