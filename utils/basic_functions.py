import awkward as ak


def add_fields(collection, fields=None, four_vec="PtEtaPhiMLorentzVector"):
    if fields == "all":
        fields = list(collection.fields)
        for field in ["pt", "eta", "phi", "mass"]:
            if field not in fields:
                fields.append(field)
        
        # remove 2d fields
        fields=[f for f in fields if ("muon" not in f and "electron" not in f)]
    
    elif fields is None:
        fields = ["pt", "eta", "phi", "mass"]
        fields_add = [
            "pt_raw",
            "mass_raw",
            "PNetRegPtRawRes",
            "PNetRegPtRawCorr",
            "PNetRegPtRawCorrNeutrino",
            "btagPNetB",
            "index",
        ]
        for field in fields_add:
            if field in list(collection.fields):
                fields.append(field)

    if four_vec == "PtEtaPhiMLorentzVector":
        fields_dict = {field: getattr(collection, field) for field in fields}
        # remove fields with 2d
        # fields_dict = {k: v for k, v in fields_dict.items() if v.ndim == 1}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    elif four_vec == "Momentum4D":
        fields_dict = {field: getattr(collection, field) for field in fields}
        # remove fields with 2d
        # fields_dict = {k: v for k, v in fields_dict.items() if v.ndim == 1}
        collection = ak.zip(
            fields_dict,
            with_name="Momentum4D",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection




def align_by_eta(full, reduced, put_none=False):
    """
    Replace collection (e.g. Jet) elements in `full` with those from `reduced`
    wherever eta matches. Missing entries in `reduced`
    are kept from `full`.

    Both `full` and `reduced` can be jagged Arrays.
    """
    full_eta = full.eta
    reduced_eta = reduced.eta

    # broadcast: (n_events, n_full, n_reduced)
    matches = full_eta[:, :, None] == reduced_eta[:, None, :]

    # for each full object, does a match exist?
    has_match = ak.any(matches, axis=2)

    # index of the matching object in reduced (if exists)
    idx = ak.argmax(matches, axis=2)

    # gather rescaled objects
    gathered = reduced[idx]

    if put_none:
        # mask out objects that had no match and put None
        aligned = ak.mask(gathered, has_match)
    else:
        # use rescaled if present, otherwise original
        aligned = ak.where(has_match, gathered, full)

    return aligned