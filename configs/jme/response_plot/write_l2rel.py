import json
import sys

# sys.path.append("../")
from configs.jme.params.binning import *


def create_pol_string(num_params):
    pol_string = "[0]"
    for i in range(1, num_params - 2):
        pol_string += f"+[{i}]*pow(log10(x),{i})"
    return pol_string


def write_l2rel_txt(main_dir, correct_eta_bins, year, num_params, version, split15, flavs):
    for flav_group in flavs:
        # create string for flavour
        flav_group_str = ""
        for flav in flav_group:
            flav_group_str += flav.replace("_", "")
        for flav in flav_group:
            flav_str=f"_{flav}Flav" if flav != "inclusive" else ""

            # create txt file for L2Relative
            file_names = [
                f"{year}_{version}_MC_L2Relative_AK4PFPNet{flav_str}.txt",
                f"{year}_{version}_MC_L2Relative_AK4PFPNetPlusNeutrino{flav_str}.txt",
            ]
            for file_name in file_names:
                with open(f"{main_dir}/{file_name}", "w") as l2_file:
                    suffix = ("Neutrino" if "Neutrino" in file_name else "") + (
                        "Tot" if split15 else ""
                    )
                    l2_file.write(
                        f"{{1 JetEta 1 JetPt ({create_pol_string(num_params)})  Correction L2Relative }}\n"
                    )
                    for i in range(len(correct_eta_bins) - 1):
                        try:
                            with open(
                                f"{main_dir}/inv_median_plots_binned/fit_results_inverse_median_Response_{flav_group_str}_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}.json",
                                "r",
                            ) as f:
                                fit_results_dict = json.load(f)

                                params_string = ""
                                for param in fit_results_dict[
                                    f"{flav}_ResponsePNetReg{suffix}"
                                ]["parameters"]:
                                    params_string += "    {}".format(param)
                                for j in range(
                                    num_params
                                    - 2
                                    - len(
                                        fit_results_dict[f"{flav}_ResponsePNetReg{suffix}"][
                                            "parameters"
                                        ]
                                    )
                                ):
                                    params_string += " 0"
                                jetpt_low = "    {}".format(
                                    fit_results_dict[f"{flav}_ResponsePNetReg{suffix}"][
                                        "jet_pt"
                                    ][0]
                                )
                                jetpt_up = "    {}".format(
                                    fit_results_dict[f"{flav}_ResponsePNetReg{suffix}"][
                                        "jet_pt"
                                    ][1]
                                )
                                l2_file.write(
                                    f" {correct_eta_bins[i]} {correct_eta_bins[i+1]} {num_params}  {jetpt_low} {jetpt_up} {params_string}\n"
                                )
                        except FileNotFoundError:
                            print(
                                f"File not found for {correct_eta_bins[i]} to {correct_eta_bins[i+1]}"
                            )
                            # set parameter 0 to 1 and the rest to 0
                            params_string = ""
                            for j in range(num_params - 2):
                                if j == 0:
                                    params_string += " 1"
                                else:
                                    params_string += " 0"
                            l2_file.write(
                                f" {correct_eta_bins[i]} {correct_eta_bins[i+1]}    {num_params}  0 0 {params_string}\n"
                            )
                        except KeyError:
                            print(
                                f"No fit for {correct_eta_bins[i]} to {correct_eta_bins[i+1]}"
                            )
                            # set parameter 0 to 1 and the rest to 0
                            params_string = ""
                            for j in range(num_params - 2):
                                if j == 0:
                                    params_string += "    1"
                                else:
                                    params_string += "    0"
                            l2_file.write(
                                f" {correct_eta_bins[i]} {correct_eta_bins[i+1]}    {num_params}  0 0 {params_string}\n"
                            )


if __name__ == "__main__":
    write_l2rel_txt(
        "/work/mmalucch/out_jme/out_cartesian_full_correctNeutrinosSeparation_jetpt_ZerosPtResponse/",
        eta_bins,
    )
