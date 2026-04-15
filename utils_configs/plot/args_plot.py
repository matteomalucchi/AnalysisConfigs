import argparse


parser = argparse.ArgumentParser(description="Plot 2b morphed vs 4b data")
parser.add_argument(
    "-i",
    "--input-data",
    type=str,
    nargs="+",
    required=True,
    help="Input directory for data with coffea files or coffea files themselves",
)
parser.add_argument(
    "-im",
    "--input-mc",
    type=str,
    nargs="+",
    help="Input coffea files monte carlo",
    default=None,
)
parser.add_argument("-o", "--output", type=str, help="Output directory", default="")
parser.add_argument(
    "-n",
    "--normalisation",
    type=str,
    help="Type of normalisation (num_events, sum_weights, density)",
    default="sum_weights",
)
parser.add_argument(
    "-sn",
    "--separate-normalisation",
    action="store_true",
    help="If true, normalise signal region and control region separately. Otherwise, normalise everything to control region.",
    default=False,
)
parser.add_argument(
    "-om",
    "--onnx-model",
    type=str,
    help="Path to the onnx containing the DNN model for SvB",
    default="",
)
parser.add_argument(
    "-r",
    "--region-suffix",
    type=str,
    help="Suffix for the region",
    default="",
)
parser.add_argument("-w", "--workers", type=int, default=8, help="Number of workers")
parser.add_argument(
    "-l", "--linear", action="store_true", help="Linear scale", default=False
)
parser.add_argument(
    "-t", "--test", action="store_true", help="Test on one variable", default=False
)
parser.add_argument(
    "-r2",
    "--run2",
    action="store_true",
    help="If running with Run2 method",
    default=False,
)
parser.add_argument(
    "-s",
    "--spread",
    action="store_true",
    help="Perform the spread morphing plot of the DNN score",
    default=False,
)
parser.add_argument(
    "-c",
    "--comparison",
    action="store_true",
    help="Compare distributions for DATA and MC and for Run2 and SPANet",
    default=False,
)
parser.add_argument(
    "-v",
    "--input-variables",
    type=str,
    help="Input variables to the onnx model",
    default="sig_bkg_dnn_input_variables",
)
parser.add_argument(
    "--novars",
    action="store_true",
    help="If true, old save format without saved variations is expected",
    default=False,
)
parser.add_argument(
    "--boosted",
    action="store_true",
    help="If true, boosted regions format is expected",
    default=False,
)

args = parser.parse_args()

print("Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
