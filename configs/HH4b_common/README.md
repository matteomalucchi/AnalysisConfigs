# HH4b analysis

> [!IMPORTANT]
> Work in Progress

This folder contains the configuration files and customization code for the HH4b analysis.

## Full analysis workflow

The full analysis workflow is composed by multiple steps, which are spread in different repositories:

- Configurations for [Pocket Coffea](https://github.com/matteomalucchi/PocketCoffea)
  - <https://github.com/matteomalucchi/AnalysisConfigs>
- Configurations for [SPANet](https://github.com/matteomalucchi/SPANet)
  - <https://github.com/matteomalucchi/HH4b_SPANet>
- DNN training
  - <https://github.com/matteomalucchi/ML_pytorch>

### Build datasets

> [!TIP]
> @Tier-3/AnalysisConfigs &rarr;  [README](https://github.com/matteomalucchi/AnalysisConfigs/blob/main/README.md)

To build the datasets needed for the Analysis, run the following command on `tier-3`:

```bash
"build-datasets --cfg datasets/datasets_definitions.json -o -rs 'T[123]_(FR|IT|BE|CH|DE|US)_\w+'"
```

### Produce SPANet input files

On `tier-3`, run the following commands to produce the input files for SPANet training.

#### Run pocket-coffea to produce coffea files

> [!TIP]
> @Tier-3/AnalysisConfigs &rarr;  [README](https://github.com/matteomalucchi/AnalysisConfigs/blob/main/README.md)

```bash
cd AnalysisConfigs/configs/HH4b
# SPANet training with normal pT spectrum
run_pocket_coffea no_model HH4b_spanet_input.py <t3_run_options> <output_dir>
# SPANet training with flat pT spectrum
run_pocket_coffea pt_vary HH4b_spanet_input.py <t3_run_options> <output_dir>

# e.g.
run_pocket_coffea no_model HH4b_spanet_input.py params/t3_run_options.yaml ../../../sample_spanet/loose_MC_postEE_btagWP
```

> [!NOTE]
> It does not matter, if the config file (e.g. `no_model`) is passed with or without the `.py` ending. The script handles this automatically.

> [!NOTE]
> To run a test on a small number of files, add the `--test` flag at the **end** of the command.

#### Convert coffea files to h5 files

> [!TIP]
> @Tier-3/HH4b_SPANet &rarr;  [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

```bash
cd HH4b_SPANet/utils/dataset
python3 coffea_to_parquet.py -i <input_coffea_file> -o <output_dir> -c 4b_region
python3 parquet_to_h5.py -i <input_parquet_files> -o <output_dir> -f 0.8

# e.g. 
python3 /work/tharte/HH4b_SPANet/utils/dataset/coffea_to_parquet.py -i .//output_all.coffea -o . -c 4b_region
python3 /work/tharte/HH4b_SPANet/utils/dataset/parquet_to_h5.py -i ./*.parquet -o /scratch/tharte/166814/ -f 0.8
```

#### Copy h5 files to `lxplus`

> [!TIP]
> @Tier-3

```bash
scp -r <dir> <user>@lxplus.cern.ch:<dir>

# e.g.
scp -r loose_MC_postEE_btagWP tharte@lxplus.cern.ch:/eos/user/t/tharte/Analysis_data/spanet_samples
```

### Train and evaluate SPANet model

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Edit the option_file accordingly to the training you want to perform.

- Event info file (input parameters to SPANet):
  - Inside the folder `HH4b_SPANet/event_files/HH4b/`
- Option file:
  - Inside the folder `HH4b_SPANet/Option_files/HH4b`
  - Lines to edit:

    - ```json
      "event_info_file": "...",                
      "training_file": "...",  
      ```

    - e.g.

    ```json
      "event_info_file": "/afs/cern.ch/user/t/tharte/public/Software/HH4b_SPANet/event_files/HH4b/hh4b_5jet_btag_wp.yaml",                
      "training_file": "/eos/user/t/tharte/Analysis_data/spanet_samples/loose_MC_postEE_btagWP/output_JetGood_train.h5",  
      ```

Then run the following command on `lxplus`:

```bash
cd HH4b_SPANet/
python jobs/submit_jobs_seed.py -o <options_files/option_file.json> -c <jobs/config/config.yaml> -s <start_seed>:<end_seed> -a <"additional arguments to pass to spanet.train"> --suffix <directory_suffix> -out <output_dir>

# e.g. 
python3 ~/public/Software/HH4b_SPANet/jobs/submit_jobs_seed.py -c ~/public/Software/HH4b_SPANet/jobs/config/jet_assignment_deep_network_3d.yaml -s 100:101 -o options_files/HH4b/hh4b_5jets_ptreg_loose_300_btag_wp.json -out /eos/user/t/tharte/Analysis_data/spanet_output
```

#### Compute SPANet predictions

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Once the model is trained, compute the predictions on the input h5 files using the following command on `lxplus`:

```bash
python -m spanet.predict <path_to_spanet_output/out_seed_training_yyy/version_z> <output/file.h5> -tf </true/file/with/inputs> --gpu

app_spanet
. ~/.bashrc
env_spanet
python -m spanet.predict ./spanet_output/out_spanet_outputs/out_hh4b_5jets_ptreg_loose_300_btag_wp/out_seed_trainings_100/version_0/ predictions/spanet_hh4b_5jets_300_ptreg_loose_s100_btag_wp.h5 -tf spanet_samples/loose_MC_postEE_btagWP/output_JetGood_test.h5 --gpu 
# In case of checking the mass sculpting with data, choose a data file as -tf argument
```

#### Plot pairing efficiency and mass sculpting

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Next step is to fill in an entry in the efficiency script `HH4b_SPANet/utils/performance/efficiency_configurations.py`:

The `efficiency_configuration` script contains two dictionaries: `spanet_dict` and `true_dict`. These have to be completed with the new models:
```python
spanet_dict = {
    ...
    '<unique_identifier>': {                                                                                            
        'file': f'</path/to/file>.h5',                                                     
        'true': '<unique_identifier_of_true_file>',                                                                              
        'label': '<label_for_plot>',                                                                                                
        'color': '<color_in_plot>'},
	...
	#e.g.
    '5_jets_ptvary_btag_wp_3V00e_allklambda': {                                                                                     
 	    'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp.h5',                                                 
 	    'true': '5_jets_pt_true_wp_allklambda',                                                                                    
 	    'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',                                                                      
 	    'color': 'orangered'},
}

true_dict = {
    ...
	'<unique_identifier>': {'name': '</path/to/truefile>', 'klambda': <'preEE'/'postEE'>}  # klambda settings define, which klambdas are in the file (preEE has less klambda than postEE)
	...
    #e.g.
	'5_jets_pt_true_wp_allklambda': {'name': f"{true_dir_thierry}../spanet_samples/loose_MC_postEE_btagWP/output_JetGood_test.h5", 'klambda': 'postEE'}, 
}
```

And then run the efficiency script in an empty folder (A different environment is needed without the apptainer image):

```bash
python3 <path/to/HH4b_SPANet>/utils/performance/efficiency_studies.py -pd <output/dir/for/plots> <-k> <-d>  # -k is to separate klambdas, -d is to run on datafiles

# e.g.
env_utils 
python3 ~/public/Software/HH4b_SPANet/utils/performance/efficiency_studies.py -pd . -k # To run the mass shapes using data inputs, replace -k with -d
```

#### Convert SPANet model to ONNX

> [!TIP]
> @lxplus/HH4b_SPANet &rarr; [README](https://github.com/matteomalucchi/HH4b_SPANet/blob/main/README.md)

Converting the file to `onnx` to use it in PocketCoffea (You need again the SPANet environment from before for the prediction):

``` bash
python -m spanet.export <path_to_spanet_output/out_seed_training_yyy/version_z> <onnx_output_name.onnx> --gpu 

#e.g.
python -m spanet.export out_spanet_outputs/out_hh4b_5jets_ptvary_loose_300_btag_wp/out_seed_trainings_100/version_0/ spanet_hh4b_5jets_ptvary_loose_300_btag_5wp_s100.onnx --gpu
```

#### Copy ONNX model to `tier-3`

> [!TIP]
> @lxplus

Finally, copy the model to `tier-3` to use it in PocketCoffea:

```bash
scp <model.onnx> <user>@t3ui07.psi.ch:<dir>
```

### Apply SPANet model to data for background morphing

> [!TIP]
> @Tier-3/AnalysisConfigs

Create a config in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) and set the `"spanet"` entry of the `onnx_model_dict` to the path of the ONNX model you copied to `tier-3`.

Then run PocketCoffea with that config to produce coffea files with SPANet predictions on data files to be used for background morphing using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

> [!NOTE]
> If the columns are saved as `parquet` in a different folder (using the `save_chunks` setting), the path to the files is stored in the `config.json`.
> If one of the scripts using the columns does not find the columns, the problem could be, that this file was overwritten/is missing.

### Train DNN model for background morphing

> [!TIP]
> @Tier-3/ML_pytorch &rarr; [README](https://github.com/matteomalucchi/ML_pytorch/blob/main/README.md)

Create a config in `ML_pytorch/configs/bkg_reweighting/` and set the `data_dirs` entry to the path of the coffea files you produced in the previous step.

Then run the training using the following command:

```bash
sbatch run_20_trainings_in_4_parallel.sh <config_file> <output_folder>

# when this has finished, you can merge the results with:
cd <output_folder>
ml_onnx -i best_models -o best_models -ar -v bkg_morphing_dnn_DeltaProb_input_variables

```

The training will produce the ONNX model to be used in PocketCoffea for background morphing, as well as plots with the training history, the ROC curve and an overtraining check.

### Apply background morphing DNN to data and produce MC signal files

> [!TIP]
> @Tier-3/AnalysisConfigs

Update the config created before  in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) and set the `"bkg_morphing_dnn"` entry of the `onnx_model_dict` to the path of the ONNX model you produced in the previous step.

Then run PocketCoffea with that config to produce coffea files with the background prediction and signal samples using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

#### Plot morphed 2b vs 4b

> [!TIP]
> @Tier-3/AnalysisConfigs

To compare the morphed 2b data with the 4b data in CR and SR, run the following command:

```bash
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python AnalysisConfigs/scripts/plot_2bMorphedvs4b.py -i <input_directory> -o <output_directory> <--novars> <-r2>"
```

### Train DNN for signal / background classification

> [!TIP]
> @Tier-3/ML_pytorch &rarr; [README](https://github.com/matteomalucchi/ML_pytorch/blob/main/README.md)

Create a config in `ML_pytorch/configs/ggF_bkg_classifier/` and set the `data_dirs` entry to the path of the coffea files you produced in the previous step.

Then run the training using the following command:

```bash
sbatch run_sig_bkg_classifier.sh <config_file> <output_folder>
```

### Apply DNN to data and MC signal files

> [!TIP]
> @Tier-3/AnalysisConfigs

> [!NOTE]
> If the goal is to produce Datacards, the `quantile_transformer` has to be run before this step (See in section [Quantile transformer to obtain constant signal binning](#Quantile-transformer-to-obtain-constant-signal-binning)).

Update the config created before  in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) and set the `"sig_bkg_dnn"` entry of the `onnx_model_dict` to the path of the ONNX model you produced in the previous step.

Then run PocketCoffea with that config to produce coffea files with the background prediction and signal samples using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

#### Plot DNN score

> [!TIP]
> @Tier-3/AnalysisConfigs


```bash
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python AnalysisConfigs/scripts/plot_DNN_score.py -i <input_directory> -im <input_signal_file> -o <output_directory> <--novars>  <-r2>"
```

### Datacard production

This section describes how to produce the datacards for the final statistical analysis.

> [!IMPORTANT]
> Work in Progress

#### Quantile transformer to obtain constant signal binning

> [!TIP]
> @Tier-3/AnalysisConfigs

The quantile transformer is mainly needed for Datacard production. The idea is to compute the bin widths for the `sig_bkg_score` variables in a way, that each bin contains the same amount of MC SM signal. This can be done in two ways:

The first option is to train the DNN model for signal / background classification and then apply this model on a previously created PocketCoffea file that used the same SPANet model for the pairing. This is the recommended way:
```bash
python </path/to/script>/extract_quantile_transformer.py -i </path/to/coffeafiles>/output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea \
	--onnx-model </path/to/model/modelname>.onnx \
	--input-variables <sig_bkg_input_variable_list_name> <--novars>
	-o <output_directory (default is ./quantile_transformer)****>
```

The second option is to use the score variables that are already in the `.coffea` files. In this case, the last PocketCoffea command has to be rerun after defining the bins to get the variables for the datacards.

Due to the need to rerun PocketCoffea, this second option is Not recommended:

```bash
python scripts/extract_quantile_transformer.py -i <input_signal_file> <--novars>
```

Set the `qt_postEE`/`qt_preEE` entry in the config created before  in [`AnalysisConfigs/configs/HH4b_common/config_files`](./config_files/) to the path of the quantile transformer you produced in the previous step.

If the first option was chosen, continue with the section [Apply DNN to data and MC signal files](#Apply-DNN-to-data-and-MC-signal-files).

Otherwise, run PocketCoffea with that config to produce coffea files using the following command:

```bash
run_pocket_coffea <config_name> <config_file> <t3_run_options> <output_dir>
```

#### Produce datacards

> [!TIP]
> @Tier-3/AnalysisConfigs


To produce the datacards, we need a single `output_all.coffea` file made from all the relevant coffea outputs from the last `PocketCoffea` run:

```bash
# Inside output folder:
pocket-coffea merge-outputs -o output_all.coffea *.coffea -f
```

Then the `build_datacards.py` script can just be run like this:

```bash
python </path/to/AnalysisConfigs>/scripts/build_datacards.py -i <input_folder> -o <desired output folder>
```

#### Compute b-tag WP efficiencies

> [!TIP]
> @Tier-3/AnalysisConfigs

We need to compute the b-tag efficiencies within the phase-space where we perform our analysis. This has to be done in a region in which we do Not cut at all on b-tags. This requires to run a different config file:

```bash
cd AnalysisConfig/configs/HH4b_btagging
pocket-coffea run --cfg config_compute_befficiency_HH4b.py -e dask@T3_CH_PSI --custom-run-options <run_option_file> -o <outputfolder>
```

Using the output from that, we can then run the scrip `produceBtagEff.py`. This file needs an input file of type `output_all.coffea`. It still needs some improvement. But the core works:
Different sample groups can be defined, that are combined and use the same efficiencies. This is expected in `YAML` fromat and can be given as input parameter `-g`. The idea is, that the same file is also used to load the groups into the params for `PocketCoffea`. An example with `ttHbb` and `HH4b` groups is given in `AnalysisConfig/configs/HH4b_common/params/btagging_sampleGroups.yaml`.

```bash
python </path/to/AnalysisConfigs>/configs/HH4b_btagging/produceBtagEff.py -i <input_file> -o <desired output folder> -g <sampleGroup file (default works if script is not moved)>
```

The output is then the b-tag WP efficiency files:
```bash
btag_efficiencies_btagDeepFlavB_2022_postEE.json
btag_efficiencies_btagPNetB_2022_postEE.json
btag_efficiencies_btagRobustParTAK4B_2022_postEE.json
```
and they have to be copied to the folder:
```bash
cp btag_efficiencies*.json AnalysisConfig/configs/HH4b_common/params/btag_efficiencies_multipleWP/
```

To then validate the procedure, we need to run on the same region a corrected and an uncorrected set.
Both should then have the same normalisation.
This part is still subject to changes and might be still bugged.
```bash
# Still inside HH4b_btagging
run_pocket_coffea no_model HH4b_parton_matching_config_btagWPsf.py ../HH4b/params/t3_run_options.yaml ../../../samples_no_model_input_for_spanet/no_model_sf_btag_comparison
```

The output from that will save histograms of different kinematic variables. This could be expanded, but should show the differences well enough.
There will be two regions saved. One is called `inclusive`, which only contains standard variations and weights and No b-tag sf. Then there is a `inclusive_btag_sf`. This contains also the b-tag sf. The histograms from both regions can be compared and should more or less fit. All histograms should have the same summed up values within each region if considering over-/underflow bins.

> **TODO.** Write file for comparison of both regions (Notebook Matteo)

## Example commands

### Run analysis

```python
run_pocket_coffea <config_name> <config_file> <run_options> <output_dir> <--test>

# e.g.
run_pocket_coffea spanet_ptflat_rerun_matteo_transform VBF_HH4b_config.py params/t3_run_options_spanet_predict_10Gb.yaml /work/mmalucch/out_hh4b/out_transformed_DNN_score
```

### Run ggF HH4b analysis

```python
cd configs/HH4b
pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask> [!TIP] @T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py
```

### Run VBF HH4b analysis

```python
cd configs/VBF_HH4b
pocket-coffea run --cfg VBF_HH4b_test_config.py -e dask> [!TIP] @T3_CH_PSI --custom-run-options params/t3_run_options_spanet_predict.yaml -o /work/mmalucch/out_hh4b/out_vbf_jets_candidates/  --executor-custom-setup onnx_executor.py
```

### Run plot 2bvs4b

```python
sbatch -p short --account=t3 --time=00:05:00 --mem 25gb --cpus-per-task=8 --wrap="python plot_2bMorphedvs4b.py -i <input_directory> -o <output_directory>"
```

### Run plot DNN_score

```python
sbatch -p short --account=t3 --time=00:10:00 --mem 40gb --cpus-per-task=1 --wrap="python ~/AnalysisConfigs/scripts/plot_DNN_score.py -id ./  -im output_GluGlutoHHto4B_spanet_kl-1p00_kt-1p00_c2-0p00_2022_postEE.coffea -r2 -om /work/mmalucch/out_ML_pytorch/DNN_DHH_method_class_weights_e5drop75_postEE_allklambda_matteo/state_dict/model_best_epoch_19.onnx"
```

### Plot variables before datacard
```bash
pocket-coffea make-plots -i output_all.coffea --cfg parameters_dump.yaml -o plots
```

### Run Datacard creation
```bash
pocket-coffea merge-outputs -o output_all.coffea *.coffea -f
python /work/tharte/datasets/AnalysisConfigs_develop/scripts/build_datacards.py -i ./ -o datacards
```
