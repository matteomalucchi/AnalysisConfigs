import argparse
import json
import logging
import os

from coffea.util import load
from pocket_coffea.utils.stat import (
    Datacard,
    DataProcess,
    DataProcesses,
    MCProcess,
    MCProcesses,
    Systematics,
    SystematicUncertainty,
)
from pocket_coffea.utils.stat.combine import combine_datacards  # , create_scripts

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger()


parser = argparse.ArgumentParser(description="Build datacards from pocket-coffea outputs")
parser.add_argument(
    "-i",
    "--input-data",
    type=str,
    nargs="+",
    required=True,
    help="Input directory for data with coffea files or coffea files themselves",
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default=""
)
args = parser.parse_args()

input_dir = os.path.dirname(args.input_data[0])

if not os.path.exists(args.output):
    os.makedirs(args.output)

coffea_list = [file for file in os.listdir(input_dir) if file.endswith(".coffea")]
if "output_all.coffea" in coffea_list:
    coffea_file = os.path.join(input_dir, "output_all.coffea")
else:
    raise NameError(f"No combined coffea file found in {coffea_list}")

# -- Load Coffea file and config.json --
coffea_file = load(coffea_file)

config_json_path = os.path.join(os.path.dirname(input_dir), "config.json")
with open(config_json_path, "r") as f:
    config = json.load(f)

# -- Histograms --
histograms_dict = {
        # "SoB": coffea_file["variables"]["sig_bkg_dnn_score_transformed"]
        "SoB": coffea_file["variables"]["sig_bkg_dnn_score"],
        "SoB_transformed": coffea_file["variables"]["sig_bkg_dnn_score_transformed"]
        }

# -- Create Processes
meta_dict_mc = {"samples": [], "years": []}
meta_dict_data = {"samples": [], "years": []}

for name, file in config["datasets"]["filesets"].items():
    metadata = file["metadata"]
    meta_dict = meta_dict_mc if metadata["isMC"] == "True" else meta_dict_data  # The boolean is a string...

    if metadata["sample"] not in meta_dict["samples"]:
        meta_dict["samples"].append(metadata["sample"])
    if metadata["year"] not in meta_dict["years"]:
        meta_dict["years"].append(metadata["year"])

logger.info(f"These are the found MC samples: {meta_dict_mc['samples']} and years {meta_dict_mc['years']}")
logger.info(f"These are the found Data samples: {meta_dict_data['samples']} and years {meta_dict_data['years']}")


mc_process = MCProcess(
        name="GluGlutoHHto4b",
        samples=meta_dict_mc["samples"],
        years=meta_dict_mc["years"],
        is_signal=True,
        )
mc_processes = MCProcesses([mc_process])

data_process = DataProcess(
        name="JetMET_JMENano",
        samples=meta_dict_data["samples"],
        # years=meta_dict_data["years"],
        )
data_processes = DataProcesses([data_process])

# -- Systematics --
common_systematics = [
    "JES_Total_AK4PFPuppi", "JER_AK4PFPuppi"
]

systematics = []
for syst in common_systematics:
    for year in meta_dict_mc["years"]:
        systematics.append(SystematicUncertainty(name=syst, datacard_name=f"{syst}_{year}", typ="shape", processes=["GluGlutoHHto4b"], years=[year], value=1.0))
systematics = Systematics(systematics)

datacard = Datacard(
        histograms=histograms_dict["SoB"],
        datasets_metadata=coffea_file["datasets_metadata"],
        cutflow=coffea_file["cutflow"],
        systematics=systematics,
        years=meta_dict_mc["years"] + meta_dict_data["years"],
        mc_processes=mc_processes,
        data_processes=data_processes,
        category="4b_signal_region",

        )

combine_datacards(
        datacards={
            "4b_signal_region": datacard
            },
        directory=args.output,
        )
