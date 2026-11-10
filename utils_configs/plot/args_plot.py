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
# For Boosted only: direction of the background morphing in the ABCD plane, same as the
# `bkg_morphing_direction` config option
morphing_direction_group = parser.add_mutually_exclusive_group()
morphing_direction_group.add_argument(
    "--bbtag-direction",
    action="store_const",
    const="btag",
    dest="morphing_direction",
    help="For Boosted only: the morphing goes along the TXbb (trained A -> C, applied to B). This is the default.",
)
morphing_direction_group.add_argument(
    "--mass-direction",
    action="store_const",
    const="mass",
    dest="morphing_direction",
    help="For Boosted only: the morphing goes along the Higgs mass (trained A -> B, applied to C).",
)
parser.set_defaults(morphing_direction="btag")
parser.add_argument(
    "-ir",
    "--input-rew",
    type=str,
    nargs="+",
    help="Input coffea files reweighted",
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
    "-cm",
    "--compare-morphing-models",
    type=str,
    nargs=2,
    metavar=("AVERAGE_ONNX", "MEDIAN_ONNX"),
    default=None,
    help=(
        "Boosted only: compare two background-morphing ONNX models, the mean "
        "over the k-folds of the weight and the median over the k-folds of the "
        "weight (average_morphing_model_kfold.onnx and "
        "median_morphing_model_kfold.onnx). Both are evaluated again on the "
        "saved morphing inputs of the postW regions (the inputs are given with "
        "--input-variables, a name in configs/HH4b_common/dnn_input_variables.py, e.g. bkg_morphing_boosted_dnn_input_variables_other_group_txbb_wp). "
        "The ttbar MC of the target region is added to both reweighted curves. "
        "Only the comparison plots are made. Unless -o is given, they are saved "
        "in <directory of the models>/plots_avg_vs_med_<normalisation>."
    ),
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
    "-ov",
    "--only-vars",
    type=str,
    nargs="+",
    default=None,
    help=(
    "For debugging: only read/plot columns whose name contains one of "
    "these substrings ('weight' and '*_N' are always kept). Greatly "
    "speeds up loading when checking a single variable. Alternatively, "
    "pass a single path ending in '.py' pointing to a script defining "
    "a dict called `dnn_input_variables` (e.g. "
    "bkg_morphing_dnn_boosted_bdt_inputs.py); its values will be joined "
    "with '_' to build the list of substrings automatically."
    ),
)
parser.add_argument(
    "--boosted",
    action="store_true",
    help="If true, boosted regions format is expected",
    default=False,
)
parser.add_argument(
    "--skip-signal-region",
    action="store_true",
    help=(
        "Boosted only: do not make the SR and SR_TT plots (data of the signal region D and the corresponding "
        "reweighted regions): only the CR and CR_TT plots are made and the signal-region data are not loaded"
    ),
    default=False,
)
parser.add_argument(
    "--mixed",
    action="store_true",
    help="mixed datasets",
    default=False,
)
parser.add_argument(
    "-cms",
    "--cmstext",
    type=str,
    help="Text shown on CMS plot",
    default="Private",
)
parser.add_argument(
    "-mw",
    "--mask-large-weights",
    action="store_true",
    help="Removes events with large weights from the plots (w>100)",
    default=True,
)

args = parser.parse_args()

print("Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
