# MC Truth corrections for PNet pT regression

Repository to compute MC Truth corrections for PNet regressed pT jets, structured as an analysis configurations for a specific development branch from a fork [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea/tree/main).

## Setup

### lxplus
To setup a local installation on `lxplus`:
```bash
# Clone the fork and checkout the desired branch
git clone --branch jme-pnet-reg https://github.com/matteomalucchi/PocketCoffea.git
cd PocketCoffea

#Enter the Singularity image
apptainer shell --bind /afs -B /cvmfs/cms.cern.ch \
         --bind /tmp  --bind /eos/cms/ -B /etc/sysconfig/ngbauth-submit \
         -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"  \
         /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable


# Create a local virtual environment using the packages defined in the apptainer image
python -m venv --system-site-packages pocket_coffea_env

# Activate the environment
source pocket_coffea_env/bin/activate

# Install in EDITABLE mode
pip install -e .[dev]

cd ../AnalysisConfigs
pip install -e .
```

After that you should set an alias to activate the PocketCoffea environment because this is called automatically by the `exec.py` script. 

On `lxplus`, it can be done by adding the following line to your `~/.bashrc`:

```bash
alias pocket_coffea='apptainer shell --bind /afs -B /cvmfs/cms.cern.ch \
         --bind /tmp  --bind /eos/cms/ -B /etc/sysconfig/ngbauth-submit \
         -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"  \
         /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable'
```

### Other systems

If instead you are using a different system, where for example you want to install the environment in micromamba, you can do the following:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -p /work/$USER/micromamba # or  ./bin/micromamba shell init -s bash -r ~/micromamba
source ~/.bashrc
micromamba create -n pocket-coffea python=3.11 -c conda-forge
micromamba activate pocket-coffea
pip install coffea==0.7.20

# Clone the fork and checkout the desired branch
git clone --branch jme-pnet-reg https://github.com/matteomalucchi/PocketCoffea.git
cd PocketCoffea
# For developers
pip install -e .[dev,docs]

cd ../AnalysisConfigs
pip install -e .
```

After that you should set an alias to activate the PocketCoffea environment because this is called automatically by the `exec.py` script.
On your system, it can be done by adding the following line to your `~/.bashrc`:

```bash
alias pocket_coffea='micromamba activate pocket-coffea'
```

## Activate the environment
### lxplus
To activate the environment, you can use the alias defined above:

```bash
source PocketCoffea/pocket_coffea_env/bin/activate
export PYTHONPATH=$PWD/PocketCoffea:$PYTHONPATH
```

### Other systems
To activate the environment, you can use the alias defined above:

```bash
pocket_coffea
```


## Workflow
### Running the analysis
To run this over the full dataset for a particular year in each $\eta$ and $p_T$ bin, you can use the following command:

```bash
python exec.py --full -pnet --dir <dir_name> -y <year> [--lxplus]
```

Where `<dir_name>` is the name of the directory where you want to save the results, and `<year>` is the year you want to run the analysis for. The `--lxplus` flag is used to indicate that you are running this on `lxplus` and it will use the `pocket_coffea_env` environment.

Year can be set to:

- 2022_preEE
- 2022_postEE
- 2023_preBPix
- 2023_postBPix

This will save the results in the `dir_name` directory inside the
`output_all.coffea` file. If running on `lxplus`, there will be an output file for each worker in the `dir_name` directory and you can merge them using:
```bash
cd <dir_name>
pocket-coffea merge-outputs -o output_all.coffea output_job_*.coffea
```

The output file contains 2D histograms for each $\eta$ bin in which the x-axis is the jet $p_T$ response and the y-axis is the jet $p_T$.


### Computing the MC Truth corrections
After running the full dataset, in order to compute the MC Truth corrections, you can use the following command:

```bash
cd response_plot/
python response.py --full -d <dir_name> --histo 
```

To run on SLURM on tier3:

```bash
cd response_plot/
sbatch -p short --account=t3 --time=00:10:00 --mem 15gb --cpus-per-task=32 --wrap="python response.py --full -d  <dir_name> --histo -n 32"
```

This will:

- Compute the median of the response in each bin in $\eta$ as a function of $p_T$.
- Get the inverse of the median.
- Fit the inverse of the median with a 6th order polynomial.
- Save the results in the configuration file.

It will also:

- Plot the histograms of the response in each bin in $\eta$ and $p_T$ bin.
- Plot the median of the response in each bin in $\eta$ as a function of $p_T$.
- Plot the inverse of the median in each bin in $\eta$ as a function of $p_T$.
- Plot the resolution of the response in each bin in $\eta$ as a function of $p_T$ using 3 different definitions.


### Closure test
To run the closure test of the corrections you can re-run the analysis with some additional flags:
```bash
python exec.py --full -pnet --dir <dir_name> -y <year> --closure --abs-eta-inclusive [--lxplus]
```
This will run the analysis applying the newly derived corrections which have to be specified in the config file. 
Once this is done, you can run the other steps of the anlaysis to obtain the final plots.

To plot all eta bins on the same plot you can use the following command:

```bash
cd response_plot/
python plot_summary_reponse.py -d <dir_name>
```
This is useful to plot the closure test of the MC Truth corrections in a inclusive way.


