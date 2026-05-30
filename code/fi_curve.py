"""
fi_curve.py
===========
Discount-curve loading and manipulation for the
UniCredit Variable Rate Bond 2034 (ISIN IT0005599110).
Depends on: fi_calendar

Exports
-------
load_curve(path) -> pd.DataFrame
make_df_fn(curve, spot_date) -> Callable[[date], float]
shift_curve(curve, bump_cc) -> pd.DataFrame
shift_curve_at_tenor(curve, tenor_y, bump_cc, width_y) -> pd.DataFrame

Curve-shifting overview
-----------------------
The base curve (Interp_term_structure.xlsx) is a daily continuously-
compounded zero-rate grid. Sensitivities are computed by shifting this
grid directly — no re-bootstrapping is required.

  shift_curve(curve, bump_cc)
      Parallel bump of all zero rates (scalar) or a per-row vector bump.
      Use for parallel DV01 and PCA factor shifts where the eigenvector
      is interpolated onto the daily TTM grid.

  shift_curve_at_tenor(curve, tenor_y, bump_cc, width_y=0.5)
      Triangular bump centred on one tenor, tapering to zero over ±width_y.
      Use for the per-tenor IRS DV01 vector, where one IRS tenor is
      shocked in isolation while the rest of the curve stays flat.
"""

from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Curve I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_curve(path: str) -> pd.DataFrame:
    """
    Load the daily-interpolated term structure from Interp_term_structure.xlsx.

    Expected columns (exact names):
        Days | TTM | Zero rate (c.c.) | DF
    Days = calendar days from the spot date; day 0 = spot.

    Returns a DataFrame sorted by Days with integer Days column.
    Raises ValueError if the column names do not match.
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    expected = ["Days", "TTM", "Zero rate (c.c.)", "DF"]
    if list(df.columns[:4]) != expected:
        raise ValueError(
            f"Unexpected columns: {df.columns.tolist()}.\n"
            f"Expected: {expected}"
        )
    df = df[expected].copy()
    df["Days"] = df["Days"].astype(int)
    return df.sort_values("Days").reset_index(drop=True)


def make_df_fn(curve: pd.DataFrame, spot_date: date) -> Callable[[date], float]:
    """
    Return a closure  df_fn(target_date: date) -> float.

    The interpolation arrays are captured once at construction time, so
    repeated calls inside a repricing loop are fast (no DataFrame lookups).

    Behaviour:
    - Returns NaN for dates strictly before spot.
    - Linearly interpolates within the grid; extrapolates flat beyond the end.

    This is the universal interface accepted by fi_bond.price_bond and
    fi_credit.price_risky_bond. Build one closure from the base curve for
    the base case; build another from a shifted curve for each sensitivity run.
    """
    days_arr = curve["Days"].values
    df_arr   = curve["DF"].values

    def df_fn(target_date: date) -> float:
        day = (target_date - spot_date).days
        if day < 0:
            return float("nan")
        return float(np.interp(day, days_arr, df_arr))

    return df_fn


# ─────────────────────────────────────────────────────────────────────────────
# Curve shifting
# ─────────────────────────────────────────────────────────────────────────────

def shift_curve(
    curve:   pd.DataFrame,
    bump_cc: "float | np.ndarray",
) -> pd.DataFrame:
    """
    Bump continuously-compounded zero rates and recompute discount factors.

    Parameters
    ----------
    curve   : base curve from load_curve()
    bump_cc : scalar  → uniform parallel shift (e.g. 1e-4 for +1 bp)
              ndarray → per-row vector, same length as curve rows

    DFs are recomputed as  P = exp(-r_bumped * TTM).
    The spot row (TTM = 0) is always kept at DF = 1.
    The original DataFrame is unchanged; a new copy is returned.

    Example — PCA factor shift
    ---------------------------
    PCA eigenvectors are defined at 8 IRS tenor nodes. To apply a factor
    shift to the full daily grid, interpolate the shift vector first:

        pca_tenors = np.array([1, 2, 3, 5, 7, 10, 15, 20])
        shift_bp = alpha * np.sqrt(lam_k) * eigenvectors[:, k]   # bp
        bump_daily = np.interp(curve['TTM'], pca_tenors, shift_bp)
        curve_shifted = shift_curve(curve, bump_daily / 1e4)      # decimal
    """
    out      = curve.copy()
    r_bumped = out["Zero rate (c.c.)"].values + bump_cc
    ttm      = out["TTM"].values
    out["DF"] = np.where(ttm > 0, np.exp(-r_bumped * ttm), 1.0)
    return out


def shift_curve_at_tenor(
    curve:   pd.DataFrame,
    tenor_y: float,
    bump_cc: float,
    width_y: float = 0.5,
) -> pd.DataFrame:
    """
    Bump one tenor and taper linearly to zero over ±width_y years.

    Used for the per-tenor IRS DV01 vector. For each IRS tenor j,
    call this function, build make_df_fn, reprice the bond, central-difference.

    Parameters
    ----------
    tenor_y : target maturity in years (e.g. 1.0, 2.0, 5.0, 7.0, 10.0)
    bump_cc : bump size in decimal (e.g. +1e-4 for +1 bp)
    width_y : half-width of the triangular taper in years (default 0.5)

    Bump profile:  weight(T) = max(0,  1 - |T - tenor_y| / width_y)
    Full bump at T = tenor_y; zero bump at T = tenor_y ± width_y.
    """
    out    = curve.copy()
    ttm    = out["TTM"].values
    weight = np.maximum(0.0, 1.0 - np.abs(ttm - tenor_y) / width_y)
    r_bumped = out["Zero rate (c.c.)"].values + bump_cc * weight
    out["DF"] = np.where(ttm > 0, np.exp(-r_bumped * ttm), 1.0)
    return out


def shift_curve_key_rate(
    curve:      "pd.DataFrame",
    tenor_y:    float,
    bump_cc:    float,
    all_tenors: "np.ndarray | None" = None,
) -> "pd.DataFrame":
    """
    Key-rate DV01 bump: hat function centred on tenor_y, tapering to zero
    at the midpoints to the adjacent tenor nodes.

    This is the standard construction for key-rate duration / DV01 vectors
    (Tuckman & Serrat, Ch. 6).  It differs from shift_curve_at_tenor in that
    the taper widths are derived from the spacing to neighbouring tenor nodes
    rather than a fixed symmetric width, so every tenor's region covers exactly
    the interval between adjacent midpoints and no two regions overlap.

    Parameters
    ----------
    curve      : base curve from load_curve()
    tenor_y    : the IRS tenor node to bump (e.g. 5.0, 7.0, 10.0)
    bump_cc    : bump size in decimal (e.g. +1e-4 for +1 bp)
    all_tenors : array of all IRS tenor nodes in ascending order.
                 Defaults to the standard 8-node grid
                 [1, 2, 3, 5, 7, 10, 15, 20].

    Returns a new DataFrame; the original is unchanged.
    """
    import numpy as np

    if all_tenors is None:
        all_tenors = np.array([1., 2., 3., 5., 7., 10., 15., 20.])

    all_tenors = np.asarray(all_tenors, dtype=float)
    idx = int(np.searchsorted(all_tenors, tenor_y))
    if idx >= len(all_tenors) or all_tenors[idx] != tenor_y:
        raise ValueError(f"{tenor_y} not found in all_tenors={all_tenors}")

    # Left and right boundaries of this tenor's region
    lo = (all_tenors[idx - 1] + tenor_y) / 2 if idx > 0 else 0.0
    hi = ((tenor_y + all_tenors[idx + 1]) / 2
          if idx < len(all_tenors) - 1
          else tenor_y + (tenor_y - all_tenors[idx - 1]))

    out  = curve.copy()
    ttm  = out["TTM"].values
    wlo  = tenor_y - lo   # left half-width
    whi  = hi - tenor_y   # right half-width

    # Triangular hat: rises linearly from lo to tenor_y, falls to hi
    left_mask  = (ttm >= lo)  & (ttm <= tenor_y)
    right_mask = (ttm >  tenor_y) & (ttm <= hi)

    weight = np.zeros_like(ttm)
    weight[left_mask]  = (ttm[left_mask]  - lo) / wlo  if wlo > 0 else 1.0
    weight[right_mask] = (hi - ttm[right_mask]) / whi  if whi > 0 else 0.0

    r_bumped = out["Zero rate (c.c.)"].values + bump_cc * weight
    out["DF"] = np.where(ttm > 0, np.exp(-r_bumped * ttm), 1.0)
    return out
