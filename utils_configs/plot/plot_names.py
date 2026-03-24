def plot_regions_names(cat_plot_name, namesuffix=""):
    cat_plot_name = cat_plot_name.replace("MC", "")
    cat_plot_name = cat_plot_name.replace("_transform", "")
    if "SPREAD" in cat_plot_name:
        cat_plot_name = cat_plot_name.split("_SPREAD")[0]
    cat_plot_name = cat_plot_name.replace("control_region", "CR")
    cat_plot_name = cat_plot_name.replace("signal_region", "SR")
    cat_plot_name = cat_plot_name.replace("_", " ")

    if "Run2" in cat_plot_name:
        cat_plot_name = cat_plot_name.replace("Run2", "")
        namesuffix += r" $D_{HH}$"
    else:
        namesuffix += " SPANet"
    cat_plot_name += namesuffix
    return cat_plot_name


# TODO: add a function to plot the columns names
def plot_columns_names():
    return
