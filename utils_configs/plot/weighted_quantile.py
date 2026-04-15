import numpy as np

def weighted_quantile(values, quantiles, weights=None, values_sorted=False, old_style=False):
    """
    Compute the weighted quantile(s) of a 1D array.
    
    Parameters
    ----------
    values : np.ndarray
        Data values.
    quantiles : float or array-like
        Quantiles to compute (range between 0 and 1).
    weights : np.ndarray, optional
        Weights for each data point. Must be same length as `values`.
    values_sorted : bool, optional
        If True, skips sorting step (assumes `values` are already sorted).
    old_style : bool, optional
        If True, uses the definition of quantile used in np.percentile pre-1.22.
    
    Returns
    -------
    quantile : float or np.ndarray
        The computed quantile(s).
    """
    values = np.array(values)
    quantiles = np.atleast_1d(quantiles)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.array(weights)

    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    weighted_cdf = np.cumsum(weights)
    weighted_cdf /= weighted_cdf[-1]

    if old_style:
        # To be consistent with numpy < 1.22
        return np.interp(quantiles, weighted_cdf, values)
    else:
        # Newer definition similar to numpy >= 1.22
        return np.interp(quantiles, weighted_cdf - 0.5 * weights / weights.sum(), values)
