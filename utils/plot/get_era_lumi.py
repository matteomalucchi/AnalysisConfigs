def get_era_lumi(dataset_data):
    era_lumi_dict = {
        "22 Era C": 4.95,
        "22 Era D": 2.92,
        "22 Era E": 5.79,
        "22 Era F": 17.6,
        "22 Era G": 2.88,
        "23 Era Cv1": 4.43,
        "23 Era Cv2": 1.28,
        "23 Era Cv3": 1.57,
        "23 Era Cv4": 10.68,
        "23 Era Dv1": 7.83,
        "23 Era Dv2": 1.67,
        "22 preEE": 4.95+2.92,
        "22 postEE": 5.79+17.6+2.88
    }
    # GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE
    era_list = []
    for dataset in dataset_data:
        if "2022" in dataset:
            if "EraC" in dataset:
                era_list.append("22 Era C")
            elif "EraD" in dataset:
                era_list.append("22 Era D")
            elif "EraE" in dataset:
                era_list.append("22 Era E")
            elif "EraF" in dataset:
                era_list.append("22 Era F")
            elif "EraG" in dataset:
                era_list.append("22 Era G")
            elif "preEE" in dataset:
                era_list.append("22 preEE")
            elif "postEE" in dataset:
                era_list.append("22 postEE")
            else:
                print("2022 data, but not identified")
        elif "2023" in dataset:
            if "EraCv1" in dataset:
                era_list.append("23 Era Cv1")
            elif "EraCv2" in dataset:
                era_list.append("23 Era Cv2")
            elif "EraCv3" in dataset:
                era_list.append("23 Era Cv3")
            elif "EraCv4" in dataset:
                era_list.append("23 Era Cv4")
            elif "EraDv1" in dataset:
                era_list.append("23 Era Dv1")
            elif "EraDv1" in dataset:
                era_list.append("23 Era Dv1")
            elif "preBPix" in dataset:
                era_list.append("23 preBPix")
            elif "postBPix" in dataset:
                era_list.append("23 postBPix")
            else:
                print("2023 data, but not identified")
    print("Found eras in datasets")
    print(era_list)
    assert len(era_list) > 0
    lumi = sum([era_lumi_dict[era] for era in era_list])
    #convert lumi to string with 2 digits
    lumi = "{:.2f}".format(lumi)
    
    # If nothing else will be satisfied:
    era_string = ", ".join(era_list)
    # If only one Era
    if len(era_list) == 1:
        era_string = era_list[0]
    # If not a full year is taken
    elif all([era in era_list for era in ["22 Era C", "22 Era D"]]):
        era_string = "22 preEE"
    elif all([era in era_list for era in ["22 Era E", "22 Era F", "22 Era G"]]):
        era_string = "22 postEE"
    elif all([era in era_list for era in ["23 Era Cv1", "22 Era Cv2"]]):
        era_string = "23 preParkingHH"
    elif all(
        [
            era in era_list
            for era in ["23 Era Cv3", "23 Era Cv4", "23 Era Dv1", "23 Era Dv2"]
        ]
    ):
        era_string = "23 postParkingHH"
    # If full years were taken
    if all(
        [
            era in era_list
            for era in ["22 Era C", "22 Era D", "22 Era E", "22 Era F", "22 Era G"]
        ]
    ):
        era_string = "2022"
    elif all(
        [
            era in era_list
            for era in [
                "23 Era Cv1",
                "23 Era Cv2",
                "23 Era Cv3",
                "23 Era Cv4",
                "23 Era Dv1",
                "23 Era Dv2",
            ]
        ]
    ):
        era_string = "2023"
    elif all([era in era_list for era in era_lumi_dict]):
        era_string = "2022, 2023"
    return lumi, era_string
