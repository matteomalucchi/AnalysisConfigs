import re

import awkward as ak
import correctionlib
import numpy as np
from pocket_coffea.lib.weights.weights import WeightData, WeightDataMultiVariation, WeightLambda, WeightWrapper


class SF_btag_fixed_multiple_wp(WeightWrapper):
    name = "sf_btag_fixed_multiple_wp"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.params = params
        self.metadata = metadata
        self._variations = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]["comb"]  # + params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]["light"]

    def compute(self, events, size, shape_variation):

        if shape_variation == "nominal":
            nominal, variations, up_var, down_var = get_sf_btag_fixed_multiple_wp(
                self.params,
                events.JetGood,
                self.metadata["year"],
                self.metadata["sample"],
                return_variations=True
            )
            return WeightDataMultiVariation(
                name=self.name,
                nominal=nominal,
                variations=variations,
                up=up_var,
                down=down_var
            )
        else:
            return WeightData(
                name=self.name,
                nominal=get_sf_btag_fixed_multiple_wp(
                    self.params,
                    events.JetGood,
                    self.metadata["year"],
                    self.metadata["sample"],
                    return_variations=False
                )
            )


SF_btag_fixed_multiple_wp_lamb = WeightLambda.wrap_func(
    name="sf_btag_fixed_multiple_wp_lamb",
    function=lambda params, metadata, events, size, shape_variations:
            get_sf_btag_fixed_multiple_wp(params, events.JetGood, metadata["year"], metadata["sample"], events.event),
    has_variations=True
    )


def get_sf_btag_fixed_multiple_wp(params, Jets, year, sample, return_variations=True):

    sampleGroups = params["btagging"]["sampleGroups"]
    Jets = Jets[:, :5]

    btag_effi_sample_group = ""
    for sampleGroupName, sampleGroup in sampleGroups.items():
        if sample in sampleGroup["sampleNames"]:
            btag_effi_sample_group = sampleGroupName
    if btag_effi_sample_group == "":
        print("WARNING: Sample does not correspond to one of the given sample groupings!")

    paramsBtagSf = params["jet_scale_factors"]["btagSF"][year]
    btag_algo = params["btagging"]["working_point"][year]["btagging_algorithm"]
    btag_wps = params["btagging"]["working_point"][year]["btagging_WP"][btag_algo]
    sf_file = paramsBtagSf["file"]
    btag_effi_file = paramsBtagSf["btagEfficiencyFile"][btag_algo]

    btag_effi_corr_set = correctionlib.CorrectionSet.from_file(btag_effi_file)
    btag_sf_corr_set = correctionlib.CorrectionSet.from_file(sf_file)

    jetpt = ak.flatten(Jets["pt"])
    jeteta = ak.flatten(Jets["eta"])
    jetflav = ak.flatten(Jets["hadronFlavour"])
    jetcounts = ak.num(Jets["pt"])
    btag_sf_name = paramsBtagSf["fixed_wp_name_map"][btag_algo]

    if return_variations:
        paramsBtagVar = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][year]

        heavy_variations = paramsBtagVar["comb"]
        light_variations = paramsBtagVar["light"]

        variation_names = []
        if heavy_variations is None:
            heavyVarUp, heavyVarDown = [], []
        if len(heavy_variations) <= 1:
            heavyVarUp = ["up"]
            heavyVarDown = ["down"]
            variation_names.append("heavyFlavor")
        else:
            heavyVarUp = ["up_" + var for var in heavy_variations]
            heavyVarDown = ["down_" + var for var in heavy_variations]
            variation_names += list(heavy_variations)

        if light_variations is None:
            lightVarUp, lightVarDown = [], []
        if len(light_variations) <= 1:
            lightVarUp = ["up"]
            lightVarDown = ["down"]
            variation_names.append("lightFlavor")
        else:
            lightVarUp = ["up_" + var for var in light_variations]
            lightVarDown = ["down_" + var for var in light_variations]
            variation_names += ["lightFlavor_" + var for var in light_variations]
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            },
            "up": {
                "light": ["central" for var in heavyVarUp] + lightVarUp,
                "heavy": heavyVarUp + ["central" for var in lightVarUp],
                "btag_sf": []
            },
            "down": {
                "light": ["central" for var in heavyVarDown] + lightVarDown,
                "heavy": heavyVarDown + ["central" for var in lightVarDown],
                "btag_sf": []
            }
        }
    else:
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            }
        }

    ''' 
    This code works for 3 working points. For more WPs, additional terms need to be added to the formula. For more details check out the BTV POG twiki https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#b-tagging-efficiencies-in-simulation
    Note: Fixed for variable WP. Just enter the WPs in following list
    '''
    wp_list = ["0", "L", "M", "T", "XT", "XXT", "1"] # These are the bin edges, I am considering. 0: btag-score <=0, 1:btag-score>1
    # Hack fake WPs into btag_wps:
    btag_wps["0"] = 0.0
    btag_wps["1"] = 1.1 # I don't want to miss a btag score of exactly 1. Might be unnecessary...
    b_wp_bins = {} #This is a dictionary I will fill with bins, that have the names of the lower edges of the bins.

    # Get all efficiencies for all wps.
    eff = {}
    for wp in wp_list:
        if wp == "0":
            eff[wp] = ak.ones_like(jetpt)
        elif wp == "1":
            eff[wp] = ak.zeros_like(jetpt)
            # eff[wp] = ak.unflatten(ak.zeros_like(jetpt), counts=jetcounts)
        else:
            eff[wp] = btag_effi_corr_set[btag_effi_sample_group + "_wp_" + wp].evaluate(jetpt, jeteta, jetflav)


    # Preloading some things.
    ones_array = ak.ones_like(jetflav)
    zero_array = ak.zeros_like(jetflav)
    mask_light = jetflav < 4
    for wp_low, wp_high in zip(wp_list[:-1], wp_list[1:]): # Always left and right edge
        # This part is inefficient. Need to find out why...
        wp = wp_low  # I name my bins after the left edge of the bins
        b_wp_bins[wp] = {}
        # Filter, which jets are belonging into this bin:
        bin_mask = (Jets[btag_algo] > btag_wps[wp_low]) & (Jets[btag_algo] <= btag_wps[wp_high])
        # Get all efficiencies into dict
        b_wp_bins[wp]["eff_left"] = ak.unflatten(eff[wp_low], counts=jetcounts)[bin_mask]
        b_wp_bins[wp]["eff_right"] = ak.unflatten(eff[wp_high], counts=jetcounts)[bin_mask]
        # SF are now a bit tricky. They depend on variationType, and the specific light and heavy variations
        b_wp_bins[wp]["sf_x_eff_left"] = {}
        b_wp_bins[wp]["sf_x_eff_right"] = {}
        for variationType, variationColl in variationsDict.items():
            b_wp_bins[wp]["sf_x_eff_left"][variationType] = {}
            b_wp_bins[wp]["sf_x_eff_right"][variationType] = {}
            # Ok so we are essentially looping over all variations of either light or heavy, and for each where one is variated,
            # the other is kept at "central"
            for variation_light, variation_heavy in zip(variationColl["light"], variationColl["heavy"]):
                # I did not find a better solution, than always calculating left and right edge. This means I repeat each operation once when it is the left and once when it is the right bin...
                for leftright, wp_tag in zip(["left", "right"], [wp_low, wp_high]):
                    if wp_tag == "0":
                        sf_flat = ones_array  # Only oneses
                        # sf_light_flat = zero_array  # Only oneses
                        # sf_heavy_flat = zero_array  # Only oneses
                    elif wp_tag == "1":
                        sf_flat = zero_array  # Only oneses
                        # sf_light_flat = zero_array  # Only zeros
                        # sf_heavy_flat = zero_array  # Only zeros
                    else:
                        # This is an ugly hack. the two evaluation fuctions fail, if wrong flavor is given.
                        # I set a mask for light flavors and set all heavy flavors as if they are heavy and vice versa.
                        # These values will not be used in the end, but otherwise, the calculation fails...
                        jetflav_light = ak.where(mask_light, jetflav, 0)
                        jetflav_heavy = ak.where(~mask_light, jetflav, 5)
                        sf_flat = ak.where(mask_light,
                                btag_sf_corr_set[btag_sf_name + "_light"].evaluate(
                                    variation_light,
                                    wp_tag,
                                    jetflav_light,
                                    abs(jeteta),
                                    jetpt
                                ), btag_sf_corr_set[btag_sf_name + "_comb"].evaluate(
                                    variation_heavy,
                                    wp_tag,
                                    jetflav_heavy,
                                    abs(jeteta),
                                    jetpt
                                )
                        )
                    b_wp_bins[wp][f"sf_x_eff_{leftright}"][variationType][f"{variation_light}_{variation_heavy}"] = ak.unflatten(sf_flat * eff[wp_tag], counts=jetcounts)[bin_mask]

    for variationType, variationColl in variationsDict.items():
        for i, (variation_light, variation_heavy) in enumerate(zip(variationColl["light"], variationColl["heavy"])):

            # Iterate through Jets distributed in bins, and lower and upper edges which are WPs.

            # Denominator: MC efficiencies, wp dependent
            # 0: L, 1: M, 2: T, 3: XT, 4: XXT
            # Make list of efficiencies:
            numerator_terms = []
            denominator_terms = []
            for wp in wp_list[:-1]:  # Here we look at the bins, and they are defined by left edge
                numerator_terms.append(
                    b_wp_bins[wp]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"] -
                    b_wp_bins[wp]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"]
                    )
                denominator_terms.append(
                    b_wp_bins[wp]["eff_left"] - b_wp_bins[wp]["eff_right"]
                    )
            # concatenate, to sum up the different WP contributions over the same events
            # prod over axis=1 is multiplying within event. Length should be the same as len(jetcounts)
            numerator = ak.prod(ak.concatenate(numerator_terms, axis=1), axis=1)
            denominator = ak.prod(ak.concatenate(denominator_terms, axis=1), axis=1)

            btag_weight_wp = numerator / denominator
            # Search for inf or None and set the weight to 1
            # btag_weight_wp = ak.fill_none(btag_weight_wp, 1, axis=1)
            if ak.any(ak.is_none(btag_weight_wp)) or ak.any(btag_weight_wp > 1000) or ak.any(btag_weight_wp < -1000):
                bad_idx = ak.is_none(btag_weight_wp) | (btag_weight_wp > 1000) | (btag_weight_wp < -1000)
                sf_x_eff_left_0 = b_wp_bins["0"]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_left_L = b_wp_bins["L"]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_left_M = b_wp_bins["M"]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_left_T = b_wp_bins["T"]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_left_XT = b_wp_bins["XT"]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_left_XXT = b_wp_bins["XXT"]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_right_0 = b_wp_bins["0"]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_right_L = b_wp_bins["L"]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_right_M = b_wp_bins["M"]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_right_T = b_wp_bins["T"]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_right_XT = b_wp_bins["XT"]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                sf_x_eff_right_XXT = b_wp_bins["XXT"]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"][bad_idx]
                raise ValueError(
                    f"Unsensible value(s): "
                    f"btag_weight={btag_weight_wp[bad_idx]}, "
                    f"b-tag score={Jets['btagPNetB'][bad_idx]}, "
                    f"pt={ak.unflatten(jetpt, counts=jetcounts)[bad_idx]}, "
                    f"eta={ak.unflatten(jeteta, counts=jetcounts)[bad_idx]}, "
                    f"flavour={ak.unflatten(jetflav, counts=jetcounts)[bad_idx]} "
                    f"b-tag_eff_left_0={b_wp_bins['0']['eff_left'][bad_idx]}, "
                    f"b-tag_eff_left_L={b_wp_bins['L']['eff_left'][bad_idx]}, "
                    f"b-tag_eff_left_M={b_wp_bins['M']['eff_left'][bad_idx]}, "
                    f"b-tag_eff_left_T={b_wp_bins['T']['eff_left'][bad_idx]}, "
                    f"b-tag_eff_left_XT={b_wp_bins['XT']['eff_left'][bad_idx]}, "
                    f"b-tag_eff_left_XXT={b_wp_bins['XXT']['eff_left'][bad_idx]}, "
                    f"b-tag_eff_right_0={b_wp_bins['0']['eff_right'][bad_idx]}, "
                    f"b-tag_eff_right_L={b_wp_bins['L']['eff_right'][bad_idx]}, "
                    f"b-tag_eff_right_M={b_wp_bins['M']['eff_right'][bad_idx]}, "
                    f"b-tag_eff_right_T={b_wp_bins['T']['eff_right'][bad_idx]}, "
                    f"b-tag_eff_right_XT={b_wp_bins['XT']['eff_right'][bad_idx]}, "
                    f"b-tag_eff_right_XXT={b_wp_bins['XXT']['eff_right'][bad_idx]}, "
                    f"sf x eff left 0={sf_x_eff_left_0}, "
                    f"sf x eff left L={sf_x_eff_left_L}, "
                    f"sf x eff left M={sf_x_eff_left_M}, "
                    f"sf x eff left T={sf_x_eff_left_T}, "
                    f"sf x eff left XT={sf_x_eff_left_XT}, "
                    f"sf x eff left XXT={sf_x_eff_left_XXT}, "
                    f"sf x eff right 0={sf_x_eff_right_0}, "
                    f"sf x eff right L={sf_x_eff_right_L}, "
                    f"sf x eff right M={sf_x_eff_right_M}, "
                    f"sf x eff right T={sf_x_eff_right_T}, "
                    f"sf x eff right XT={sf_x_eff_right_XT}, "
                    f"sf x eff right XXT={sf_x_eff_right_XXT}, "
                    f"in variation {variationType}, "
                    f"light {variation_light}, heavy {variation_heavy}"
                )

            variationsDict[variationType]["btag_sf"].append(btag_weight_wp)
    if return_variations:
        return (variationsDict["central"]["btag_sf"][0],
                variation_names,
                variationsDict["up"]["btag_sf"],
                variationsDict["down"]["btag_sf"])
    return variationsDict["central"]["btag_sf"][0]


def sf_btag_fixed_multiple_wp_calibrated(events, params, Jets, year, sample, njets, jetsHt, return_variations=True):
    """Warning: This function is outdated and might not be working. Might be deleted soon."""
    sampleGroups = params["btagging"]["sampleGroups"]

    btag_effi_sample_group = ""
    for sampleGroupName, sampleGroup in sampleGroups.items():
        if sample in sampleGroup["sampleNames"]:
            btag_effi_sample_group = sampleGroupName
    if btag_effi_sample_group == "":
        print("WARNING: Sample does not correspond to one of the given sample groupings!")

    paramsBtagSf = params["jet_scale_factors"]["btagSF"][year]
    btag_algo = params["btagging"]["working_point"][year]["btagging_algorithm"]
    btag_wps = params["btagging"]["working_point"][year]["btagging_WP"]
    sf_file = paramsBtagSf["file"]
    btag_effi_file = paramsBtagSf["btagEfficiencyFile"][btag_algo]
    cset_calib = correctionlib.CorrectionSet.from_file(
        params.btagSF_calibration[year]["file"]
    )
    corr_calib = cset_calib[params.btagSF_calibration[year]["name"]]

    btag_effi_corr_set = correctionlib.CorrectionSet.from_file(btag_effi_file)
    btag_sf_corr_set = correctionlib.CorrectionSet.from_file(sf_file)

    jetpt = ak.flatten(Jets["pt"])
    jeteta = ak.flatten(Jets["eta"])
    jetflav = ak.flatten(Jets["hadronFlavour"])
    jetcounts = ak.num(Jets["pt"])
    njets = ak.to_numpy(njets)
    jetsHt = ak.to_numpy(jetsHt)

    jetpt_heavy = ak.flatten(Jets["pt"][Jets["hadronFlavour"] > 3])
    jeteta_heavy = ak.flatten(Jets["eta"][Jets["hadronFlavour"] > 3])
    jetflav_heavy = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"] > 3])

    jetpt_light = ak.flatten(Jets["pt"][Jets["hadronFlavour"] < 4])
    jeteta_light = ak.flatten(Jets["eta"][Jets["hadronFlavour"] < 4])
    jetflav_light = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"] < 4])

    if return_variations:
        paramsBtagVar = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][year]
        btag_sf_name = paramsBtagSf["fixed_wp_name_map"][btag_algo]

        heavy_variations = paramsBtagVar["comb"]
        light_variations = paramsBtagVar["light"]

        variation_names = []
        if heavy_variations is None:
            heavyVarUp, heavyVarDown = [], []
        if len(heavy_variations) <= 1:
            heavyVarUp = ["up"]
            heavyVarDown = ["down"]
            variation_names.append("heavyFlavor")
        else:
            heavyVarUp = ["up_" + var for var in heavy_variations]
            heavyVarDown = ["down_" + var for var in heavy_variations]
            variation_names += list(heavy_variations)

        if light_variations is None:
            lightVarUp, lightVarDown = [], []
        if len(light_variations) <= 1:
            lightVarUp = ["up"]
            lightVarDown = ["down"]
            variation_names.append("lightFlavor")
        else:
            lightVarUp = ["up_" + var for var in light_variations]
            lightVarDown = ["down_" + var for var in light_variations]
            variation_names += ["lightFlavor_" + var for var in light_variations]

        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            },
            "up": {
                "light": ["central" for var in heavyVarUp] + lightVarUp,
                "heavy": heavyVarUp + ["central" for var in lightVarUp],
                "btag_sf": []
            },
            "down": {
                "light": ["central" for var in heavyVarDown] + lightVarDown,
                "heavy": heavyVarDown + ["central" for var in lightVarDown],
                "btag_sf": []
            }
        }
    else:
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            }
        }

    ''' 
    This code works for 3 working points. For more WPs, additional terms need to be added to the formula. For more details check out the BTV POG twiki https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#b-tagging-efficiencies-in-simulation
    '''
    wp_list = ["L", "M", "T"]
    effi_MC = {}
    sf_x_eff_tagged = {}
    Jets_tagged = {}
    Jets_not_tagged = {}

    print(list(btag_effi_corr_set.keys()))
    for wp in wp_list:
        # Storing btag efficiencies for MC (taken from the maps previously computed) in different branches of Jets, one for each working point
        effi_MC[wp] = ak.unflatten(
        btag_effi_corr_set[btag_effi_sample_group + "_wp_" + wp].evaluate(jetpt, jeteta, jetflav),
        counts=jetcounts
    )
        Jets = ak.with_field(Jets, effi_MC[wp], "effi_MC_" + wp)
        sf_x_eff_tagged[wp] = {}

        for variationType, variationColl in variationsDict.items():
            sf_x_eff_tagged[wp][variationType] = []
            for variation_light, variation_heavy in zip(variationColl["light"], variationColl["heavy"]):
                sf_light_flat = btag_sf_corr_set[btag_sf_name + "_light"].evaluate(
                    variation_light,
                    wp,
                    jetflav_light,
                    abs(jeteta_light),
                    jetpt_light
                )
                sf_heavy_flat = btag_sf_corr_set[btag_sf_name + "_comb"].evaluate(
                    variation_heavy,
                    wp,
                    jetflav_heavy,
                    abs(jeteta_heavy),
                    jetpt_heavy
                )
                sf_flat = ak.to_numpy(ak.copy(jetpt))
                sf_flat[jetflav > 3] = sf_heavy_flat
                sf_flat[jetflav < 4] = sf_light_flat
                sf_DATA_MC = ak.unflatten(sf_flat, jetcounts)

                pass_wp = Jets[btag_algo] > btag_wps[wp]
                Jets_tagged[wp] = Jets[pass_wp]
                Jets_not_tagged[wp] = Jets[~pass_wp]

                effi_data = ak.prod([sf_DATA_MC, effi_MC[wp]], axis=0)

                # Lists of dictionaries containing the efficiencies for the events that passed/didn't pass the WP selection for different wps and variation type (central,up,down)
                sf_x_eff_tagged[wp][variationType].append(effi_data[pass_wp])

    for variationType, variationColl in variationsDict.items():
        for i, (variation_light, variation_heavy) in enumerate(zip(variationColl["light"], variationColl["heavy"])):

            # Note this code is done for multiple WP -> adapt the formula for more wps

            # Denominator: MC efficiencies, wp dependent
            p_MC_1 = Jets_tagged[wp_list[-1]]["effi_MC_" + wp_list[-1]]
            p_MC_2 = 1 - Jets_not_tagged[wp_list[0]]["effi_MC_" + wp_list[0]]
            eff_MC_tagged_L = ak.firsts(Jets_tagged[wp_list[0]]["effi_MC_" + wp_list[0]])
            eff_MC_not_tagged_M = ak.firsts(Jets_not_tagged[wp_list[1]]["effi_MC_" + wp_list[1]])
            eff_MC_not_tagged_T = ak.firsts(Jets_not_tagged[wp_list[-1]]["effi_MC_" + wp_list[-1]])
            eff_MC_tagged_M = ak.firsts(Jets_tagged[wp_list[1]]["effi_MC_" + wp_list[1]])

            p_MC_3 = [[x] for x in (eff_MC_tagged_L - eff_MC_not_tagged_M)]
            p_MC_4 = [[x] for x in (eff_MC_tagged_M - eff_MC_not_tagged_T)]

            # Numerator: DATA efficiencies, wp and variation dependent
            p_DATA_1 = sf_x_eff_tagged[wp_list[-1]][variationType][i]
            #   p_DATA_2 = 1 - sf_x_eff_untagged[wp_list[0]][variationType][i]
            #   sf_x_eff_thisvar_tagged_L = ak.firsts(sf_x_eff_tagged[wp_list[0]][variationType][i])
            #   sf_x_eff_thisvar_not_tagged_M = ak.firsts(sf_x_eff_untagged[wp_list[1]][variationType][i])
            #   sf_x_eff_thisvar_not_tagged_T = ak.firsts(sf_x_eff_untagged[wp_list[-1]][variationType][i])
            sf_x_eff_thisvar_tagged_M = ak.firsts(sf_x_eff_tagged[wp_list[1]][variationType][i])

            p_DATA_3 = [[x] for x in (sf_x_eff_thisvar_tagged_L - sf_x_eff_thisvar_not_tagged_M)]
            p_DATA_4 = [[x] for x in (sf_x_eff_thisvar_tagged_M - sf_x_eff_thisvar_not_tagged_T)]

            p_MC = ak.concatenate([p_MC_1, p_MC_2, p_MC_3, p_MC_4], axis=1)
            p_DATA = ak.concatenate([p_DATA_1, p_DATA_2, p_DATA_3, p_DATA_4], axis=1)

            p_MC = ak.prod(p_MC, axis=-1)
            p_DATA = ak.prod(p_DATA, axis=-1)
            btag_sf_fixed_wp = np.divide(p_DATA, p_MC)

            # Search for inf or None and set the weight to 1
            finite_mask = np.isfinite(ak.to_numpy(btag_sf_fixed_wp))
            btag_sf_fixed_wp = ak.where(finite_mask, btag_sf_fixed_wp, 1)

            var = re.sub(r'^(up|down)_', '', variation_heavy)

            # do not apply calibration on ttbar semilep -> dedicated function "ttbar_split_calibration" later that splits in subsamples
            if sample.startswith("TTToLNu2Q"):
                ones = np.ones(len(btag_sf_fixed_wp))
                variationsDict[variationType]["btag_sf"].append(ones)

            else:
                if variationType == "central":
                    output = btag_sf_fixed_wp * corr_calib.evaluate(sample,
                                                            "nominal",
                                                            njets, jetsHt)

                if (variationType == "up") & (var != "central"):
                    output = btag_sf_fixed_wp * corr_calib.evaluate(sample, f"sf_btag_fixed_multiple_wp_{var}Up", njets, jetsHt)
                elif (variationType == "down") & (var != "central"):
                    output = btag_sf_fixed_wp * corr_calib.evaluate(sample, f"sf_btag_fixed_multiple_wp_{var}Down", njets, jetsHt)

                variationsDict[variationType]["btag_sf"].append(output)

    if return_variations:
        return (variationsDict["central"]["btag_sf"][0],
                variation_names,
                variationsDict["up"]["btag_sf"],
                variationsDict["down"]["btag_sf"])
    return variationsDict["central"]["btag_sf"][0]


class SF_btag_calibration(WeightWrapper):
    name = "sf_btag_multiplewp_calib"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.params = params
        self.metadata = metadata
        self._variations = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]["comb"]
        self.jet_coll = params.jet_scale_factors.jet_collection.btag

    def compute(self, events, size, shape_variation):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)

        if shape_variation == "nominal":
            nominal, variations, up_var, down_var = sf_btag_fixed_multiple_wp_calibrated(
                events,
                self.params,
                events.JetGood,
                self.metadata["year"],
                self.metadata["sample"],
                events[f"n{self.jet_coll}"],
                jetsHt,
                return_variations=True
            )
            return WeightDataMultiVariation(
                name=self.name,
                nominal=nominal,
                variations=variations,
                up=up_var,
                down=down_var
            )
        else:
            return WeightData(
                name=self.name,
                nominal=sf_btag_fixed_multiple_wp_calibrated(
                    events,
                    self.params,
                    events.JetGood,
                    self.metadata["year"],
                    self.metadata["sample"],
                    events[f"n{self.jet_coll}"],
                    jetsHt,
                    return_variations=False
                )
            )
