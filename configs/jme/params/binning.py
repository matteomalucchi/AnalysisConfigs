import numpy as np
import os

eta_bins = [
    -5.191,
    -4.889,
    -4.716,
    -4.538,
    -4.363,
    -4.191,
    -4.013,
    -3.839,
    -3.664,
    -3.489,
    -3.314,
    -3.139,
    -2.964,
    -2.853,
    -2.65,
    -2.5,
    -2.322,
    -2.172,
    -2.043,
    -1.93,
    -1.83,
    -1.74,
    -1.653,
    -1.566,
    -1.479,
    -1.392,
    -1.305,
    -1.218,
    -1.131,
    -1.044,
    -0.957,
    -0.879,
    -0.783,
    -0.696,
    -0.609,
    -0.522,
    -0.435,
    -0.348,
    -0.261,
    -0.174,
    -0.087,
    0.000,
    0.087,
    0.174,
    0.261,
    0.348,
    0.435,
    0.522,
    0.609,
    0.696,
    0.783,
    0.879,
    0.957,
    1.044,
    1.131,
    1.218,
    1.305,
    1.392,
    1.479,
    1.566,
    1.653,
    1.74,
    1.83,
    1.93,
    2.043,
    2.172,
    2.322,
    2.5,
    2.65,
    2.853,
    2.964,
    3.139,
    3.314,
    3.489,
    3.664,
    3.839,
    4.013,
    4.191,
    4.363,
    4.538,
    4.716,
    4.889,
    5.191,
]

eta_sign_dict = {
    "neg1": [-5.191, -3.314],
    "neg2": [-3.314, -1.83],
    "neg3": [-1.83, -0.957],
    "neg4": [-0.957, 0.0],
    "pos1": [0.0, 0.957],
    "pos2": [0.957, 1.83],
    "pos3": [1.83, 3.314],
    "pos4": [3.314, 5.191],
    # "all": [-5.191, 5.191],
}

for eta_sign, eta_interval in eta_sign_dict.items():
    if str(os.environ.get("SIGN", None)) == eta_sign:
        eta_bins = [
            i for i in eta_bins if i >= eta_interval[0] and i <= eta_interval[1]
        ]
        break

inclusive_bins = [0.0, 1.3, 2.4, 2.7, 3.0, 5.0]
if int(os.environ.get("ABS_ETA_INCLUSIVE", 0)) == 1:
    eta_bins = inclusive_bins

central_bins = [-5.191, -1.3, 1.3, 5.191]
if int(os.environ.get("CENTRAL", 0)) == 1:
    eta_bins = central_bins

print("eta_bins: ", eta_bins)

pt_bins_all = (
    [
        15.0,
        17.0,
        20.0,
        23.0,
        27.0,
        30.0,
        35.0,
        40.0,
        45.0,
        57.0,
        72.0,
        90.0,
        120.0,
        150.0,
        200.0,
        300.0,
        400.0,
        550.0,
        750.0,
        1000.0,
        1500.0,
        2000.0,
        2500.0,
        3000.0,
        3500.0,
        4000.0,
        4500.0,
        5000.0,
    ]
)
pt_bins_reduced= [
        50.0,
        57.0,
        72.0,
        90.0,
        120.0,
        150.0,
        200.0,
        300.0,
        400.0,
        550.0,
        750.0,
        1000.0,
        1500.0,
        2000.0,
        2500.0,
        3000.0,
        3500.0,
        4000.0,
        4500.0,
        5000.0,
    ]

if (int(os.environ.get("PNETREG15", 0)) == 1 or int(os.environ.get("SPLITPNETREG15", 0)) == 1):
    pt_bins=pt_bins_all
else:
    pt_bins=pt_bins_reduced

print("pt_bins: ", pt_bins)


# response_bins = [0, 0.8] + list(np.arange(0.9, 1.2, 0.1)) + [1.2, 8.0]
# response_bins = [0, 0.8] + list(np.arange(0.8004, 1.2, 4e-4)) + [1.2, 8.0]
# response_bins = [0, 0.6] + list(np.arange(0.6008, 1.4, 8e-4)) + [1.4, 8.0]
response_bins = list(np.linspace(0, 8, 16000))

jet_pt_bins = list(np.linspace(0, 6000, 16000))
