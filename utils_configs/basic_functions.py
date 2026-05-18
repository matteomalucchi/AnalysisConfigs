import awkward as ak
import numpy as np
import vector
vector.register_awkward()


def add_fields(collection, fields=None, four_vec="PtEtaPhiMLorentzVector"):
    if fields == "all":
        fields = list(collection.fields)
        for field in ["pt", "eta", "phi", "mass"]:
            if field not in fields:
                fields.append(field)

        # remove 3d fields
        fields = [f for f in fields if getattr(collection, f).ndim <= 2]

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

    if not isinstance(fields, list):
        raise ValueError("fields must be a list of fields or 'all' or None")

    if four_vec == "PtEtaPhiMLorentzVector":
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    elif four_vec == "Momentum4D":
        fields_dict = {field: getattr(collection, field) for field in fields}
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


def legendre(l: int, x):
    """
    Evaluate P_l(x) for array x using the recurrence relation.
    """
    x = np.asarray(x, dtype=float)
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x.copy()         # shouldn't be able to change x

    P_km2 = np.ones_like(x)     # P_0
    P_km1 = x                   # P_1 — no copy needed, loop only reads x

    for k in range(1, l):
        P_k   = ((2*k + 1) * x * P_km1 - k * P_km2) / (k + 1)
        P_km2 = P_km1
        P_km1 = P_k

    return P_km1


def compute_fw_momenta(self, jet_collection="JetGood", l_max=4, scheme="W_T"):
    """
    Vectorised Fox-Wolfram moments for all events at once.
    Returns arrays of shape (n_events, l_max+1).

    Available weighting schemes for Fox-Wolfram momenta.
        W_s  : |p_i||p_j| / (Sum p_i)^2                                classic e+e- (CMS energy normalisation)
        W_p  : |p_i||p_j| / (Sum |p_i|)^2                              total 3-momentum normalisation
        W_T  : p_Ti p_Tj  / (Sum p_Ti)^2                               transverse momentum (hadron colliders)
        W_z  : p_zi p_zj  / (Sum p_zi)^2                               longitudinal component
        W_y  : 1/(|y_i-y_mean|*|y_j-y_mean|) / (Sum 1/|y_i-y_mean|)^2  rapidity-centred
        W_1  : 1 / n^2                                                 uniform (unweighted)
    """
    jets     = self.events[jet_collection]
    n_events = len(jets)
    max_jets = int(ak.max(ak.num(jets, axis=1)))

    # This is needed to vectorise the FW Momenta calculation and avoid looping over the events
    # Pad jagged -> rectangular (n_events, max_jets), fill missing with 0
    def pad(arr):
        return ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    arr,
                    max_jets,
                    axis=1,
                    clip=True
                ),
                0.0
            )
        )

    pt_pad   = pad(jets.pt)    # (n_events, max_jets)
    eta_pad  = pad(jets.eta)
    phi_pad  = pad(jets.phi)
    mass_pad = pad(jets.mass)

    # Boolean mask for real (non-padded) jets: (n_events, max_jets)
    counts = ak.to_numpy(ak.num(jets, axis=1))
    valid  = np.arange(max_jets)[None, :] < counts[:, None]

    # 3-momentum components and magnitudes
    px    = pt_pad * np.cos(phi_pad)
    py    = pt_pad * np.sin(phi_pad)
    pz    = pt_pad * np.sinh(eta_pad)
    p_mag = np.sqrt(px**2 + py**2 + pz**2)      # (n_events, max_jets)

    # Batched cos(theta) matrix via einsum: (n_events, max_jets, max_jets)
    p3        = np.stack([px, py, pz], axis=-1)

    dot       = np.einsum('eic,ejc->eij', p3, p3)   # np.einsum is a generalised einstein tensor contraction.
                                                    # The string 'eic,ejc->eij' is a notation that tells numpy which indices to sum over and which to keep.
                                                    # e — event, i — first jet index, j — second jet index, c — spatial component (x, y, z)

    mag_outer = p_mag[:, :, None] * p_mag[:, None, :] # (n_events, max_jets, 1) * (n_events, 1, max_jets) = (n_events, max_jets, max_jets) - entry [e, i, j] is |pi|⋅|pj|
    mag_outer = np.where(mag_outer > 0, mag_outer, 1.0) # avoid division by zero for zero-momentum jets (e.g. from padding) - these will be masked out later anyway, so the exact value doesn't matter as long as it's not zero
    cos_mat   = np.clip(dot / mag_outer, -1.0, 1.0) # guard against floating point values like 1.0000000002 with the clipping to [-1, 1]

    # Per-event weights (n_events, max_jets), zero for padded slots according to https://arxiv.org/pdf/1212.4436
    if scheme == "W_T":
        vals = np.where(valid, pt_pad, 0.0)
    elif scheme == "W_p":
        vals = np.where(valid, p_mag, 0.0) # scalar sum of 3-momenta, includes the longitudinal component so it's more sensitive to forward jets than W_T
    elif scheme == "W_z":
        vals = np.where(valid, np.abs(pz), 0.0) # ses only |p_z| so it's sensitive to the beam-axis structure
    elif scheme == "W_1":
        vals = np.where(valid, 1.0, 0.0) # uniform weights, just 1/n for n jets, so W_ij = 1/n^2
    elif scheme == "W_s":
        E    = np.sqrt(p_mag**2 + mass_pad**2)
        Esum = np.where(valid, E, 0.0).sum(axis=1, keepdims=True)
        Esum = np.where(Esum > 0, Esum, 1.0)
        # Already normalised: W_ij = (p_i/Esum) * (p_j/Esum)
        vals = np.where(valid, p_mag / Esum, 0.0)
    elif scheme == "W_y":
        E     = np.sqrt(p_mag**2 + mass_pad**2)
        denom = np.where(E - pz > 1e-9, E - pz, 1e-9) # guard E-pz can be ~ 0 for very forward jets
        y     = 0.5 * np.log((E + pz) / denom)
        n_v   = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        y_bar = np.where(valid, y, 0.0).sum(axis=1, keepdims=True) / n_v # mean rapidity of the valid jets in each event, Jets close to the mean rapidity y_bar get a large weight
        diff  = np.where(valid, np.abs(y - y_bar), 1.0) # distance from mean rapidity, set to 1 for padded jets which will be masked out anyway
        diff  = np.where(diff < 1e-9, 1e-9, diff) # floor for real jets sitting exactly at y_bar to avoid infinite weights, these jets will dominate the FW moments but that's physically reasonable since they are very central and should have a big impact on the event shape
        vals  = np.where(valid, 1.0 / diff, 0.0)
    else:
        raise ValueError(f"Unknown scheme '{scheme}'")

    # Normalise (W_s is already normalised by construction above)
    if scheme != "W_s":
        total = vals.sum(axis=1, keepdims=True)
        vals  = vals / np.where(total > 0, total, 1.0)

    # Weight outer product, zeroed for padded pairs: (n_events, max_jets, max_jets)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    w_mat      = np.where(pair_valid, vals[:, :, None] * vals[:, None, :], 0.0)

    # Fox-Wolfram moments — only loop over l (typically 0-4)
    cos_flat = cos_mat.reshape(n_events, -1)   # (n_events, max_jets ^2)
    w_flat   = w_mat.reshape(n_events, -1)

    H_out = np.zeros((n_events, l_max + 1))
    for l in range(l_max + 1):
        Pl          = legendre(l, cos_flat)    # (n_events, max_jets ^2)
        H_out[:, l] = (w_flat * Pl).sum(axis=1) # sum over all jet pairs for each event

    H0    = H_out[:, 0:1]
    R_out = np.where(H0 != 0, H_out / np.where(H0 != 0, H0, 1.0), H_out)

    return H_out, R_out