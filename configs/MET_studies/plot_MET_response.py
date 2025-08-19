import os
import sys
import re
from matplotlib import pyplot as plt
from coffea.util import load
from omegaconf import OmegaConf
import numpy as np
import awkward as ak
import mplhep as hep
import argparse

from utils.plot.get_columns_from_files import get_columns_from_files
from plot_config import var_name_dict


hep.style.use("CMS")


parser = argparse.ArgumentParser(description="Plot MET distributions from coffea files")
parser.add_argument (
    "-i",
    "--input-dir",
    type=str,
    required=True,
    help="Input directory for data with coffea files",
)
parser.add_argument(
    "-o", "--output", type=str, help="Output directory", default=""
)

args = parser.parse_args()

outputdir = args.output if args.output else "plots_MET_response"
# Create output directory if it does not exist
if not os.path.exists(outputdir):
    os.makedirs(outputdir)



def create_histos(qT_arr, resolutions_arr):
    # compute mean of all metrics in summary
    max_x=200 # max qT value
    x_n=20 #number of bins

    bin_edges=np.arange(0, max_x, 10)
    inds=np.digitize(qT_arr,bin_edges)
    print(inds[inds>7])
    qT_hist=[]
    for i in range(1, len(bin_edges)):
        qT_hist.append((bin_edges[i]+bin_edges[i-1])/2.)
    
    resolution_hists={}
    for key in resolutions_arr:
        print(key)
        R_arr=resolutions_arr[key][2] 
        u_perp_arr=resolutions_arr[key][0]
        u_par_arr=resolutions_arr[key][1]
        
        print(R_arr, len(R_arr))
        print(u_perp_arr, len(u_perp_arr))
        print(u_par_arr, len(u_par_arr))

        u_perp_hist=[] 
        u_perp_scaled_hist=[]
        u_par_hist=[]
        u_par_scaled_hist=[]
        R_hist=[]

        for i in range(1, len(bin_edges)):
            R_i=R_arr[np.where(inds==i)[0]]
            R_hist.append(np.mean(R_i))
            u_perp_i=u_perp_arr[np.where(inds==i)[0]]
            u_perp_scaled_i=u_perp_i/np.mean(R_i)
            u_perp_hist.append((np.quantile(u_perp_i,0.84)-np.quantile(u_perp_i,0.16))/2.)
            u_perp_scaled_hist.append((np.quantile(u_perp_scaled_i,0.84)-np.quantile(u_perp_scaled_i,0.16))/2.)
            u_par_i=u_par_arr[np.where(inds==i)[0]]
            u_par_scaled_i=u_par_i/np.mean(R_i)
            u_par_hist.append((np.quantile(u_par_i,0.84)-np.quantile(u_par_i,0.16))/2.)
            u_par_scaled_hist.append((np.quantile(u_par_scaled_i,0.84)-np.quantile(u_par_scaled_i,0.16))/2.)

        u_perp_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_perp_hist)
        u_perp_scaled_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_perp_scaled_hist)
        u_par_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_par_hist)
        u_par_scaled_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_par_scaled_hist)
        R=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=R_hist)
        resolution_hists[key] = {
            'u_perp_resolution': u_perp_resolution,
            'u_perp_scaled_resolution': u_perp_scaled_resolution,
            'u_par_resolution': u_par_resolution,
            'u_par_scaled_resolution':u_par_scaled_resolution,
            'R': R
        }
    return resolution_hists


def plot_histos(histo_dict, cat):
    for var_name in ['u_perp_resolution', 'u_perp_scaled_resolution', 'u_par_resolution', 'u_par_scaled_resolution', 'R']:
        fig, ax = plt.subplots()
        for key in histo_dict:
            ax.errorbar(histo_dict[key][var_name][1][:-1], histo_dict[key][var_name][0], label=key, fmt=".",)
        ax.legend()
        ax.set_xlabel(r'Z q$_{\mathrm{T}}$ [GeV]')
        ax.set_ylabel(var_name if var_name not in var_name_dict else var_name_dict[var_name])
        hep.cms.lumitext(r"(13.6 TeV)", ax=ax)
        hep.cms.text(text="Simulation Preliminary", ax=ax)
        fig.savefig(f'{outputdir}/{cat}_{var_name}.png')
        plt.close(fig)

def make_plots(cat_col):
    for cat in cat_col:
        print(f"Processing category: {cat}")
        col_dict = cat_col[cat]
        v_qT=col_dict["ll_pt"]

        resolutions_arr={}
        for var in col_dict:
            if "_MuonGood" in var and any(x in var for x in ["u_perp_predict", "u_paral_predict", "response"]):
                coll=var.split("_")[0]
                if coll not in resolutions_arr:
                    resolutions_arr[coll]=[]
                print(var)
                resolutions_arr[coll].append(col_dict[var])
        
        resolutions_arr = create_histos(v_qT, resolutions_arr)
        plot_histos(resolutions_arr, cat)


if __name__ == "__main__":
    
    inputfiles_data = [
        os.path.join(args.input_dir, file)
        for file in os.listdir(args.input_dir)
        if file.endswith(".coffea")
    ]
    
    cat_col, total_datasets_list = get_columns_from_files(inputfiles_data)
    print(f"Total datasets found: {total_datasets_list}")
    print(cat_col)
    
    make_plots(cat_col)
    
