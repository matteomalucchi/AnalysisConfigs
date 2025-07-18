# python exec.py --full -pnet --central --dir _correctNeutrinosSeparation_jetpt_ZerosPtResponse --neutrino 1 / 0

import subprocess
import argparse
from params.binning import eta_bins, eta_sign_dict
import os
import shutil
import sys
import shlex

parser = argparse.ArgumentParser(description="Run the jme analysis")
parser.add_argument(
    "--inclusive-eta",
    "-i",
    action="store_true",
    help="Run over eta bins",
    default=False,
)
parser.add_argument(
    "--kill", "-k", action="store_true", help="Kill all tmux sessions", default=False
)
parser.add_argument(
    "--cartesian",
    action="store_true",
    help="Run cartesian multicuts",
    default=False,
)
parser.add_argument(
    "-p", "--parallel", action="store_true", help="Run parallel eta bins", default=False
)
parser.add_argument(
    "-s",
    "--sign",
    help="Sign of eta bins",
    type=str,
    default="neg3",
)
parser.add_argument(
    "-fs",
    "--flavsplit",
    action="store_true",
    help="Flavour split",
    default=False,
)
parser.add_argument(
    "-pnet",
    "--pnet",
    action="store_true",
    help="Use ParticleNet regression",
    default=False,
)
parser.add_argument(
    "-t",
    "--test",
    action="store_true",
    help="Test run",
    default=False,
)
parser.add_argument(
    "-d",
    "--dir",
    help="Output directory",
    type=str,
    default="",
)
parser.add_argument(
    "--suffix",
    help="Suffix",
    type=str,
    default="",
)
parser.add_argument(
    "-y",
    "--year",
    help="Year",
    type=str,
    default="2023_preBPix",
)
parser.add_argument(
    "-f",
    "--flav",
    help="Flavour",
    type=str,
    default="inclusive",
)
parser.add_argument(
    "--full",
    action="store_true",
    help="Run full cartesian analysis in all eta bins and all flavours sequentially",
    default=False,
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="Make plots",
    default=False,
)
parser.add_argument(
    "--central",
    action="store_true",
    help="Central eta bin (-1.3, 1.3)",
    default=False,
)
parser.add_argument(
    "-a",
    "--abs-eta-inclusive",
    action="store_true",
    help="Run over inclusive abs eta bins",
    default=False,
)
parser.add_argument(
    "-c",
    "--closure",
    action="store_true",
    help="Produce closure test",
    default=False,
)
parser.add_argument(
    "--pnet-reg-15",
    action="store_true",
    help="Evaluate ParticleNet regression also for jet with pT < 15 GeV",
    default=False,
)
parser.add_argument(
    "--split-pnet-reg-15",
    action="store_true",
    help="Evaluate ParticleNet regression also for jet with pT < 15 GeV and slit between < and > 15 GeV",
    default=False,
)
parser.add_argument(
    "--neutrino",
    help="Sum neutrino pT to GenJet pT",
    default=-1,
    type=int,
)
parser.add_argument(
    "--lxplus",
    action="store_true",
    help="Run on lxplus",
    default=False,
)
args = parser.parse_args()

args.flavsplit = int(args.flavsplit)
args.pnet = int(args.pnet)
args.central = int(args.central)
args.closure = int(args.closure)
args.pnet_reg_15 = int(args.pnet_reg_15)
args.split_pnet_reg_15 = int(args.split_pnet_reg_15)
args.neutrino = int(args.neutrino)
args.abs_eta_inclusive = int(args.abs_eta_inclusive)

# Define a list of eta bins
eta_bins = eta_bins if not args.inclusive_eta else None

pocket_coffea_env_commands = ["pocket_coffea"]
if args.lxplus:
    env_path = os.path.dirname(
        subprocess.run(
            ["which pocket-coffea"], capture_output=True, text=True, shell=True
        ).stdout.strip()
    ).rsplit("/", 1)[0]
    pocket_coffea_env_commands.append(f"source {env_path}/bin/activate")
    pythonpath = env_path.rsplit("/", 1)[0]
    pocket_coffea_env_commands.append(f"export PYTHONPATH={pythonpath}:$PYTHONPATH")

if args.test:
    executor = "--test"
elif args.lxplus:
    run_options_file = "params/lxplus_run_options_big.tmp.yaml"
    executor = f"-e condor@lxplus --custom-run-options {run_options_file}"
else:
    run_options_file = "params/t3_run_options_big.yaml"
    executor = f"-e dask@T3_CH_PSI --custom-run-options {run_options_file}"

eta_sign_list = list(eta_sign_dict.keys())
order_eta_sign_list = ["pos1", "pos2", "pos3", "pos4", "neg4", "neg3", "neg2", "neg1"]
if len(eta_sign_list) == len(order_eta_sign_list):
    eta_sign_list = order_eta_sign_list

dir_prefix = os.environ.get("WORK", ".") + "/out_jme/"
print("dir_prefix", dir_prefix)


def run_command(sign, flav, dir_name, complete_bash_list):
    # neutrino_string = (
    #     f"&& export NEUTRINO={args.neutrino}" if args.neutrino != -1 else ""
    # )
    env_var_dict = {
        "CARTESIAN": 1,
        "SIGN": sign,
        "FLAVSPLIT": args.flavsplit,
        "PNET": args.pnet,
        "FLAV": flav,
        "CENTRAL": args.central,
        "ABS_ETA_INCLUSIVE": args.abs_eta_inclusive,
        "CLOSURE": args.closure,
        "PNETREG15": args.pnet_reg_15,
        "SPLITPNETREG15": args.split_pnet_reg_15,
        "YEAR": args.year,
    }
    if args.neutrino != -1:
        env_var_dict["NEUTRINO"] = args.neutrino
    export_string = " && ".join(
        [f"export {key}={value}" for key, value in env_var_dict.items()]
    )
    complete_bash_list.append(export_string)

    if args.lxplus and not args.test:
        # create a new run_options_file adding the environment variables

        base_run_options_file = run_options_file.replace(".tmp", "")
        
        # Start the bash script as a string, with proper escaping
        run_options_lines = [
            f'cp "{base_run_options_file}" "{run_options_file}"',
            f'echo "" >> "{run_options_file}"',
            f'echo "" >> "{run_options_file}"',
            f'echo "# Added by exec.py" >> "{run_options_file}"',
            f'echo "custom-setup-commands:" >> "{run_options_file}"'
        ]

        for key, value in env_var_dict.items():
            # Use single quotes inside echo to preserve spaces
            run_options_lines.append(
                f'echo "  - export {key}={shlex.quote(str(value))}" >> "{run_options_file}"'
            )

        complete_bash_list+= run_options_lines

    
    complete_bash_list.append(
        f"time pocket-coffea run --cfg cartesian_config.py {executor} -o {dir_name}"
    )
    if args.plot:
        complete_bash_list.append(f"make_plots.py {dir_name} --overwrite -j 16")
    

    if args.neutrino == 1:
        dir_name_no_neutrino = dir_name.replace("_neutrino", "")
        os.makedirs(dir_name_no_neutrino, exist_ok=True)
        complete_bash_list.append(
            f"cp {dir_name}/output_all.coffea {dir_name_no_neutrino}/output_all_neutrino.coffea"
        )
        complete_bash_list.append(
            f"cp {dir_name}/output_all.coffea {dir_name_no_neutrino}/output_all_neutrino.coffea"
        )
    
    return complete_bash_list


if __name__ == "__main__":
    if args.cartesian or args.full:
        complete_bash_list = ["#!/bin/bash",]
        print(
            f"Running cartesian multicuts {'in full configuration sequentially' if args.full else ''}"
        )
        sign = args.sign
        flav = args.flav

        flavs_list = (
            ["inclusive"]  # , "b", "c", "g", "uds"
            if (args.full and (args.central or args.abs_eta_inclusive))
            else ["inclusive"]
        )

        if args.full and args.neutrino != 1:
            tmux_session = "full_cartesian" + args.suffix + f"_{args.year}"
        elif args.full and args.neutrino == 1:
            tmux_session = "full_cartesian_neutrino" + args.suffix + f"_{args.year}"
        else:
            tmux_session = f"{sign}_cartesian" + args.suffix + f"_{args.year}"

        command0 = f"tmux kill-session -t {tmux_session}"
        subprocess.run(command0, shell=True)
        print(f"killed session {tmux_session}")
        if args.kill:
            sys.exit(0)
            
        command1 = f"tmux new-session -d -s {tmux_session}"
        subprocess.run(command1, shell=True)
        for env_command in pocket_coffea_env_commands:
            subprocess.run(f'tmux send-keys "{env_command}" "C-m"', shell=True)
            # complete_bash_list.append(env_command)

        eta_string = ""
        if args.abs_eta_inclusive:
            eta_string = "absinclusive"
        elif args.central:
            eta_string = "central"

        if args.full:
            for sign in (
                eta_sign_list if (not args.central and not args.abs_eta_inclusive) else [""]
            ):
                if sign == "all":
                    continue
                for flav in flavs_list:
                    dir_name = f"{dir_prefix}out_cartesian_full{args.dir}{'_pnetreg15' if args.pnet_reg_15 else ''}{'_splitpnetreg15' if args.split_pnet_reg_15 else ''}_{args.year}{'_closure' if args.closure else ''}{'_test' if args.test else ''}/{sign if not eta_string else eta_string}eta_{flav}flav{'_pnet' if args.pnet else ''}{'_neutrino' if args.neutrino == 1 else ''}"
                    if not os.path.isfile(f"{dir_name}/output_all.coffea"):
                        print(f"{dir_name}")
                        complete_bash_list=run_command(sign, flav, dir_name, complete_bash_list)
        else:
            dir_name = (
                f"{dir_prefix}out_cartesian_{sign if not eta_string else eta_string}eta{'_flavsplit' if args.flavsplit else f'_{args.flav}flav'}{'_pnet' if args.pnet else ''}{'_neutrino' if args.neutrino == 1 else ''}{args.dir}{'_pnetreg15' if args.pnet_reg_15 else ''}{'_splitpnetreg15' if args.split_pnet_reg_15 else ''}_{args.year}{'_closure' if args.closure else ''}{'_test' if args.test else ''}"
                if not args.dir
                else args.dir
            )
            if not os.path.isfile(f"{dir_name}/output_all.coffea"):
                print(f"{dir_name}")
                complete_bash_list=run_command(sign, flav, dir_name, complete_bash_list)

        complete_bash_script= "\n".join(complete_bash_list)
        # save to a file
        complete_bash_script_file=f"./run_cartesian_{tmux_session}.tmp.sh"
        with open(complete_bash_script_file, "w") as f:
            f.write(complete_bash_script)
        # make the file executable
        os.chmod(complete_bash_script_file, 0o755)
        #execute the bash script in tmux
        tmux_command = f'tmux send-keys "bash -c \'{complete_bash_script_file}\'" "C-m"'
        subprocess.run(tmux_command, shell=True)
        print(f"tmux attach -t {tmux_session}")

    else:
        # Loop over the eta bins
        if eta_bins:
            if args.parallel:
                print(f"Running over eta bins {eta_bins} in parallel")
                for i in range(len(eta_bins) - 1):
                    eta_bin_min = eta_bins[i]
                    eta_bin_max = eta_bins[i + 1]

                    comand0 = f"tmux kill-session -t {eta_bin_min}to{eta_bin_max}"
                    command1 = f'tmux new-session -d -s {eta_bin_min}to{eta_bin_max} && tmux send-keys "export ETA_MIN={eta_bin_min}" "C-m" "export ETA_MAX={eta_bin_max}" "C-m"'
                    command2 = f'tmux send-keys "pocket_coffea" "C-m" "time pocket-coffea run --cfg jme_config.py  {executor} -o out_separate_eta_bin/eta{eta_bin_min}to{eta_bin_max}" "C-m"'
                    command3 = f'tmux send-keys "make_plots.py out_separate_eta_bin/eta{eta_bin_min}to{eta_bin_max} --overwrite -j 1" "C-m"'
                    subprocess.run(comand0, shell=True)
                    print(f"killed session {eta_bin_min}to{eta_bin_max}")
                    if args.kill:
                        sys.exit(0)
                    subprocess.run(command1, shell=True)
                    subprocess.run(command2, shell=True)
                    # subprocess.run(command3, shell=True)
                    print(f"tmux attach -t {eta_bin_min}to{eta_bin_max}")
            else:
                print(f"Running over eta bins {eta_bins} in sequence")
                comand0 = f"tmux kill-session -t eta_bins"
                command1 = f"tmux new-session -d -s eta_bins"
                # execute the commands
                subprocess.run(comand0, shell=True)
                subprocess.run(command1, shell=True)

                # os.system(comand0)
                # os.system(command1)
                print(f"tmux attach -t eta_bins")
                command5 = f'tmux send-keys "pocket_coffea" "C-m"'
                subprocess.run(command5, shell=True)
                for i in range(len(eta_bins) - 1):
                    eta_bin_min = eta_bins[i]
                    eta_bin_max = eta_bins[i + 1]
                    dir_name = (
                        f"{dir_prefix}out_separate_eta_bin_seq{'_pnet' if args.pnet else ''}{'_pnetreg15' if args.pnet_reg_15 else ''}{'_splitpnetreg15' if args.split_pnet_reg_15 else ''}_{args.year}{'_closure' if args.closure else ''}{'_test' if args.test else ''}/eta{eta_bin_min}to{eta_bin_max}"
                        if not args.dir
                        else args.dir
                    )
                    command2 = f'tmux send-keys "export ETA_MIN={eta_bin_min} && export ETA_MAX={eta_bin_max} && export PNET={args.pnet}" "C-m"'
                    command3 = f'tmux send-keys "time pocket-coffea run --cfg jme_config.py  {executor} -o {dir_name}" "C-m"'
                    # command4 = f'tmux send-keys "make_plots.py {dir_name} --overwrite -j 8" "C-m"'

                    # os.environ["ETA_MIN"] = f"{eta_bin_min}"
                    # os.environ["ETA_MAX"] = f"{eta_bin_max}"

                    if not os.path.isfile(f"{dir_name}/output_all.coffea"):
                        print(f"{dir_name}")
                        subprocess.run(command2, shell=True)
                        subprocess.run(command3, shell=True)
                        # subprocess.run(comand4, shell=True)
                        # os.system(command3)
                    else:
                        print(f"{dir_name}/output_all.coffea already exists!")

        else:
            print("No eta bins defined")
            print("Running over inclusive eta")
            comand0 = f"tmux kill-session -t inclusive_eta"
            command1 = f"tmux new-session -d -s inclusive_eta"
            command2 = f'tmux send-keys "pocket_coffea" "C-m" "time pocket-coffea run --cfg jme_config.py  {executor} -o out_inclusive_eta" "C-m"'
            command3 = (
                f'tmux send-keys "make_plots.py out_inclusive_eta --overwrite -j 8" "C-m"'
            )
            subprocess.run(comand0, shell=True)
            print("killed session inclusive_eta")
            if args.kill:
                sys.exit(0)
            subprocess.run(command1, shell=True)
            subprocess.run(command2, shell=True)
            subprocess.run(command3, shell=True)
            print("tmux attach -t inclusive_eta")
