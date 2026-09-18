import re

import awkward as ak
import correctionlib
import numpy as np
import copy
from pocket_coffea.lib.weights.weights import WeightData, WeightDataMultiVariation, WeightLambda, WeightWrapper

from utils_configs.jet_algorithms import get_btag_algorithm, get_btag_working_points


class SF_btag_fixed_multiple_wp(WeightWrapper):
    name = "sf_btag_fixed_multiple_wp"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.params = params
        self.metadata = metadata
        self._varsdict = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]
        self._heavyvars = self._varsdict["heavy"][self._varsdict["heavy"]["used"]]
        self._lightvars = self._varsdict["light"][self._varsdict["light"]["used"]]
        self._variations = [f"{self._varsdict['heavy']['prefix']}_{var}" for var in self._heavyvars] + [f"{self._varsdict['light']['prefix']}_{var}" for var in self._lightvars]

    def compute(self, events, size, shape_variation):

        if shape_variation == "nominal":
            nominal, variations, up_var, down_var = self.get_sf_btag_fixed_multiple_wp(
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
                nominal=self.get_sf_btag_fixed_multiple_wp(
                    self.params,
                    events.JetGood,
                    self.metadata["year"],
                    self.metadata["sample"],
                    return_variations=False,
                    shape_variation=shape_variation
                )
            )

    def get_sf_btag_fixed_multiple_wp(self, params, Jets, year, sample, return_variations=True, shape_variation="nominal"):

        sampleGroups = params["btagging"]["sampleGroups"]
        # Additional algorithm, to correctly sort and choose jets used for analysis
        # if params["only5jetsbSF"]:
        # Jets = Jets[ak.argsort(Jets.btagB, axis=1, ascending=False)]
        # JetsHiggs = Jets[:, :4]

        # jets5plus = Jets[:, 4:]
        # jets5plus_pt = jets5plus[ak.argsort(jets5plus.pt, axis=1, ascending=False)]
        # Jets = ak.concatenate((JetsHiggs, jets5plus_pt), axis=1)
        # Jets = copy.copy(Jets[:, :5])

        btag_effi_sample_group = ""
        for sampleGroupName, sampleGroup in sampleGroups.items():
            if sample in sampleGroup["sampleNames"]:
                btag_effi_sample_group = sampleGroupName
        if btag_effi_sample_group == "":
            print("WARNING: Sample does not correspond to one of the given sample groupings!")

        paramsBtagSf = params["jet_scale_factors"]["btagSF"][year]
        btag_algo = get_btag_algorithm(params, year)
        btag_wps = get_btag_working_points(params, year, btag_algo)
        sf_file = paramsBtagSf["file"]
        btag_effi_file = paramsBtagSf["btagEfficiencyFile"][btag_algo]

        btag_effi_corr_set = correctionlib.CorrectionSet.from_file(btag_effi_file)
        btag_sf_corr_set = correctionlib.CorrectionSet.from_file(sf_file)

        jetpt = ak.flatten(Jets["pt"])
        jeteta = abs(ak.flatten(Jets["eta"]))
        jetflav = ak.flatten(Jets["hadronFlavour"])
        jetcounts = ak.num(Jets["pt"])
        btag_sf_name = paramsBtagSf["fixed_wp_name_map"][btag_algo]

        if return_variations:
            heavy_variations = self._heavyvars
            light_variations = self._lightvars

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
                variation_names += [f"{self._varsdict['heavy']['prefix']}_{var}" for var in heavy_variations]

            if light_variations is None:
                lightVarUp, lightVarDown = [], []
            if len(light_variations) <= 1:
                lightVarUp = ["up"]
                lightVarDown = ["down"]
                variation_names.append("lightFlavor")
            else:
                lightVarUp = ["up_" + var for var in light_variations]
                lightVarDown = ["down_" + var for var in light_variations]
                variation_names += [f"{self._varsdict['light']['prefix']}_{var}" for var in light_variations]
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
            else:
                eff[wp] = btag_effi_corr_set[f"{btag_effi_sample_group}_{shape_variation}_wp_{wp}"].evaluate(jetpt, abs(jeteta), jetflav)

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
                        elif wp_tag == "1":
                            sf_flat = zero_array  # Only oneses
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
                    numerator_term = b_wp_bins[wp]["sf_x_eff_left"][variationType][f"{variation_light}_{variation_heavy}"] - b_wp_bins[wp]["sf_x_eff_right"][variationType][f"{variation_light}_{variation_heavy}"]
                    denominator_term = b_wp_bins[wp]["eff_left"] - b_wp_bins[wp]["eff_right"]
                    # Updated recommendataions
                    # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/
                    # denominator_term = ak.where(numerator_term < 0, 1, denominator_term, axis=1)
                    # numerator_term = ak.where(numerator_term < 0, 1, numerator_term, axis=1)
                    numerator_terms.append(numerator_term)
                    denominator_terms.append(denominator_term)
                # concatenate, to sum up the different WP contributions over the same events
                # prod over axis=1 is multiplying within event. Length should be the same as len(jetcounts)
                numerator = ak.prod(ak.concatenate(numerator_terms, axis=1), axis=1)
                denominator = ak.prod(ak.concatenate(denominator_terms, axis=1), axis=1)

                btag_weight_wp = numerator / denominator
                if ak.any(ak.is_none(btag_weight_wp)) or ak.any(btag_weight_wp > 1000) or ak.any(btag_weight_wp < -1000):
                    # Huge error printout in case something bad is encountered.
                    # This was used, because it was difficutlt to debug properly.
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
                        f"numerator={numerator[bad_idx]}, "
                        f"denominator={denominator[bad_idx]}, "
                        f"b-tag score={Jets['btagB'][bad_idx]}, "
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


SF_btag_fixed_multiple_wp_lamb = WeightLambda.wrap_func(
    name="sf_btag_fixed_multiple_wp_lamb",
    function=lambda params, metadata, events, size, shape_variations:
            get_sf_btag_fixed_multiple_wp(params, events.JetGood, metadata["year"], metadata["sample"], events.event, shape_variation="nominal"),
    has_variations=True
    )


