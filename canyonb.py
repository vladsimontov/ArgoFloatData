"""
canyonb.py
==========
A compact, self-contained implementation of the CANYON-B prediction step
(Bittig et al. 2018, Front. Mar. Sci. 5:328, doi:10.3389/fmars.2018.00328):
Bayesian neural-network estimation of nutrient concentrations from a water
sample's date, position, pressure, temperature, salinity and oxygen. The
method maps what GO-SHIP-quality hydrography implies a sensor should read,
which makes it a self-consistency check for BGC sensors, not an independent
measurement.

This file implements the published algorithm (committee of two-hidden-layer
tanh networks with evidence weighting and the documented input transforms)
in NumPy and is licensed with this repository (MIT). The trained network
weights are NOT distributed here: they are fetched at runtime from the
authors' official repository (github.com/HCBScienceProducts/CANYON-B,
GPL-3.0) and cached on disk as data files. Cite Bittig et al. (2018) when
using predictions.

Validated against the official check value: for 2014-12-09 08:45 UTC at
17.6 N, -24.3 E, 180 dbar, 16 degC, 36.1 psu, 104 umol/kg O2, NO3 must be
17.91522 umol/kg (run `python canyonb.py` to re-verify).

Only the nutrient variables are implemented (NO3, PO4, SiOH4): they share
the no-year input layout. The carbonate-system variables would need the
year input and, for pCO2, a carbonate solver, so they are out of scope.
"""

import os

import numpy as np

WEIGHTS_URL = ("https://raw.githubusercontent.com/HCBScienceProducts/"
               "CANYON-B/master/wgts_{var}.txt")
NUTRIENTS = ("NO3", "PO4", "SiOH4")

# Arctic basin polygon ("Bering Strait prolongation") from the reference
# implementation: latitudes are shifted inside it before scaling
_ARC_LAT = np.array([68, 66.5, 66.5, 80, 80, 90, 90, 68, 68, 90, 90, 68],
                    dtype="float64")
_ARC_LON = np.array([-180, -170, -85, -80, -37, -37, 143, 143, 180, 180,
                     -180, -180], dtype="float64")


def fetch_weights(var, dest_dir):
    """Download the official weight file for `var` into dest_dir (cached).
    Returns the local path, or None if it cannot be fetched."""
    if var not in NUTRIENTS:
        return None
    dest = os.path.join(dest_dir, f"wgts_{var}.txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        import requests
        os.makedirs(dest_dir, exist_ok=True)
        r = requests.get(WEIGHTS_URL.format(var=var), timeout=60)
        r.raise_for_status()
        with open(dest + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dest + ".part", dest)
        return dest
    except Exception:
        return None


def _in_arctic(lat, lon):
    """Ray-casting point-in-polygon for the Arctic shift region."""
    inside = np.zeros(lat.shape, dtype=bool)
    n = len(_ARC_LAT)
    j = n - 1
    for i in range(n):
        yi, xi = _ARC_LAT[i], _ARC_LON[i]
        yj, xj = _ARC_LAT[j], _ARC_LON[j]
        cross = (((xi > lon) != (xj > lon))
                 & (lat < (yj - yi) * (lon - xi) / ((xj - xi) or 1e-12) + yi))
        inside ^= cross
        j = i
    return inside


def _inputs(year, lat, lon, pres, temp, psal, doxy):
    """The documented 8-column input matrix; nutrients drop the year column."""
    lat = np.asarray(lat, dtype="float64").copy()
    lon = np.asarray(lon, dtype="float64").copy()
    lon = np.where(lon > 180, lon - 360, lon)
    arc = _in_arctic(lat, lon)
    if arc.any():
        lat[arc] = lat[arc] - np.sin(np.pi * (lon[arc] + 37) / 180) \
            * (90 - lat[arc]) * 0.5
    pres = np.asarray(pres, dtype="float64")
    return np.column_stack([
        np.asarray(year, dtype="float64"),
        lat / 90.0,
        np.abs(1 - np.mod(lon - 110, 360) / 180),
        np.abs(1 - np.mod(lon - 20, 360) / 180),
        np.asarray(temp, dtype="float64"),
        np.asarray(psal, dtype="float64"),
        np.asarray(doxy, dtype="float64"),
        pres / 2e4 + 1.0 / (1 + np.exp(-pres / 300.0)) ** 3,
    ])


def predict(var, year, lat, lon, pres, temp, psal, doxy, weights_path):
    """Committee prediction of `var` (umol/kg) for 1-D sample arrays.
    Returns an array shaped like pres, NaN where inputs were not finite."""
    if var not in NUTRIENTS:
        raise ValueError(f"unsupported variable {var!r}")
    table = np.genfromtxt(weights_path)
    n_nets = table.shape[1] - 1
    norm = table[:, -1]

    full = _inputs(year, lat, lon, pres, temp, psal, doxy)
    data = full[:, 1:]                      # nutrients: no year column
    ni = data.shape[1]
    ok = np.isfinite(data).all(axis=1)
    out = np.full(len(data), np.nan)
    if not ok.any():
        return out
    mw, sw = norm[:ni + 1], norm[ni + 1:2 * ni + 2]
    x = (data[ok] - mw[:ni]) / sw[:ni]

    com_w = table[3, :n_nets]
    acc = np.zeros(x.shape[0])
    for col in range(n_nets):
        nl1 = int(table[0, col])
        nl2 = int(table[1, col])
        if nl2 <= 0:
            raise ValueError("unexpected single-hidden-layer network")
        w = table[:, col]
        p = 4
        w1 = w[p:p + nl1 * ni].reshape(ni, nl1).T       # column-major fill
        p += nl1 * ni
        b1 = w[p:p + nl1]
        p += nl1
        w2 = w[p:p + nl2 * nl1].reshape(nl1, nl2).T
        p += nl2 * nl1
        b2 = w[p:p + nl2]
        p += nl2
        w3 = w[p:p + nl2].reshape(nl2, 1).T
        p += nl2
        b3 = w[p:p + 1]
        h1 = np.tanh(x @ w1.T + b1)
        h2 = np.tanh(h1 @ w2.T + b2)
        y = (h2 @ w3.T + b3)[:, 0]
        acc += com_w[col] * y
    pred = acc / com_w.sum()
    out[ok] = pred * sw[ni] + mw[ni]        # denormalize
    return out


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    p = fetch_weights("NO3", d)
    assert p, "could not fetch weights"
    # decimal year as the reference code computes it (days/365)
    year = 2014 + (np.datetime64("2014-12-09T08:45")
                   - np.datetime64("2014-01-01T00:00")) \
        / np.timedelta64(1, "D") / 365.0
    v = predict("NO3", [year], [17.6], [-24.3], [180.0], [16.0], [36.1],
                [104.0], p)
    print(f"NO3 check: {v[0]:.5f} (expected 17.91522)")
    assert abs(v[0] - 17.91522) < 2e-3, "check value mismatch"
    print("CANYON-B self-test PASSED")


if __name__ == "__main__":
    _selftest()
