"""
fi_bond.py
==========
Bond specification constants, coupon schedule, vol surface interpolation,
Displaced Black option pricer, and the core risk-free bond pricer for the
UniCredit Variable Rate Bond 2034 (ISIN IT0005599110).

Depends on: fi_calendar, fi_curve

Exports — constants
-------------------
TRADE_DATE, SPOT_LAG, ISSUE_DATE, MATURITY_DATE
NOTIONAL, PARTICIPATION
CURRENT_COUPON_RATE, CURRENT_PERIOD_START
CAP_STRIKE_COUPON, FLOOR_STRIKE_COUPON
K_EFF_CAP, K_EFF_FLOOR
SHIFT, SCALE_CUTOFF_Y, TENOR_SCALE_3M

Exports — functions
-------------------
build_schedule(issue_date, maturity_date, holidays) -> list[dict]
load_vol_surface(path, sheet) -> pd.DataFrame
get_flat_vol(vol_surface, cap_maturity_y, strike_dec) -> float
displaced_black(F, K, T, sigma, df_pay, alpha, delta, phi) -> float
price_bond(df_fn, holidays, schedule, spot_date, sigma_cap, sigma_flr, ...) -> dict
accrued_interest(trade_date, period_start, coupon_rate, notional) -> float
"""

from __future__ import annotations

import math
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

from fi_calendar import (
    modified_following,
    subtract_business_days,
    days_30_360,
    yearfrac_act_360,
    yearfrac_act_365,
)
from fi_curve import make_df_fn  # noqa: F401 — re-exported for notebook convenience


# ─────────────────────────────────────────────────────────────────────────────
# Bond specification  (UniCredit IT0005599110)
# ─────────────────────────────────────────────────────────────────────────────

TRADE_DATE           = date(2025, 11, 5)
SPOT_LAG             = 2                      # TARGET business days
ISSUE_DATE           = date(2024, 6, 12)
MATURITY_DATE        = date(2034, 6, 12)
NOTIONAL             = 1_000.0                # EUR per bond
PARTICIPATION        = 1.60
CURRENT_COUPON_RATE  = 0.032464               # 3.2464% p.a. — fixed 10-Sep-2025 reset
CURRENT_PERIOD_START = date(2025, 9, 12)      # start of the running coupon period

CAP_STRIKE_COUPON    = 0.0545                 # 5.45% cap on the coupon rate
FLOOR_STRIKE_COUPON  = 0.0000                 # 0.00% floor on the coupon rate
K_EFF_CAP            = CAP_STRIKE_COUPON   / PARTICIPATION  # 3.40625% on EURIBOR
K_EFF_FLOOR          = FLOOR_STRIKE_COUPON / PARTICIPATION  # 0.00000% on EURIBOR

# Displaced Black model — VCAP3A surface (ICAP, 05-Nov-2025, shift = 3%)
SHIFT          = 0.03                         # displacement parameter δ
SCALE_CUTOFF_Y = 2.5                          # apply sqrt(3/6) correction above this
TENOR_SCALE_3M = math.sqrt(0.25 / 0.50)      # = 1/sqrt(2) ≈ 0.7071


# ─────────────────────────────────────────────────────────────────────────────
# Coupon schedule
# ─────────────────────────────────────────────────────────────────────────────

def build_schedule(
    issue_date:    date,
    maturity_date: date,
    holidays:      set,
) -> list:
    """
    Generate the full quarterly coupon schedule from issue to maturity.

    Steps:
      1. Unadjusted quarterly anchors starting from issue_date.
      2. Modified Following business-day adjustment.
      3. Reset date = adjusted start − 2 TARGET business days.

    Returns a list of dicts with keys:
        Period, Reset Date, Start Date, End Date, Payment Date
    """
    unadj = [issue_date]
    d = issue_date
    while d < maturity_date:
        d = d + relativedelta(months=3)
        unadj.append(d)
    adj = [modified_following(x, holidays) for x in unadj]
    periods = []
    for i in range(1, len(adj)):
        start = adj[i - 1]
        end   = adj[i]
        periods.append({
            "Period":       i,
            "Reset Date":   subtract_business_days(start, 2, holidays),
            "Start Date":   start,
            "End Date":     end,
            "Payment Date": end,
        })
    return periods


# ─────────────────────────────────────────────────────────────────────────────
# Volatility surface
# ─────────────────────────────────────────────────────────────────────────────

_ABS_STRIKES_PCT = [-1.5, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]

_TENOR_YEARS = {
    "1Y": 1.0, "18M": 1.5, "2Y":  2.0, "3Y":  3.0, "4Y":  4.0,  "5Y":  5.0,
    "6Y": 6.0, "7Y":  7.0, "8Y":  8.0, "9Y":  9.0, "10Y": 10.0, "12Y": 12.0,
    "15Y": 15.0, "20Y": 20.0, "25Y": 25.0, "30Y": 30.0,
}


def load_vol_surface(path: str, sheet: str = "6M") -> pd.DataFrame:
    """
    Parse the VCAP3A Displaced Black vol surface (ICAP, 05-Nov-2025, δ = 3%).

    Sheet layout (sheet '6M'):
      Col 0: tenor label (1Y … 30Y)
      Col 1: 6M EURIBOR ATM strike — stored for reference, not used in interp.
      Col 2: ATM flat vol (%)
      Cols 3-15: flat vols (%) at absolute strikes
                 -1.5, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0

    All vols and strikes stored as decimals in the returned DataFrame.
    Interpolation uses absolute strike — see get_flat_vol().
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    records = []
    for _, row in raw.iloc[1:].iterrows():
        tenor_str = str(row.iloc[0]).strip()
        if tenor_str not in _TENOR_YEARS:
            continue
        rec = {
            "tenor_str":   tenor_str,
            "tenor_years": _TENOR_YEARS[tenor_str],
            "atm_stk":     float(row.iloc[1]) / 100.0,
            "vol_atm":     float(row.iloc[2]) / 100.0,
        }
        for k_pct, v in zip(_ABS_STRIKES_PCT, row.iloc[3:].tolist()):
            try:
                rec[f"k_{k_pct}"] = float(v) / 100.0
            except (ValueError, TypeError):
                rec[f"k_{k_pct}"] = float("nan")
        records.append(rec)
    return pd.DataFrame(records).sort_values("tenor_years").reset_index(drop=True)


def _interp_strike(row: pd.Series, strike_dec: float) -> float:
    """Linear interpolation in the strike dimension for one tenor row.
    The ATM point is included as an additional interior knot."""
    ks = [row["atm_stk"]]
    vs = [row["vol_atm"]]
    for k_pct in _ABS_STRIKES_PCT:
        col = f"k_{k_pct}"
        if col in row.index and not np.isnan(row[col]):
            ks.append(k_pct / 100.0)
            vs.append(row[col])
    arr  = sorted(zip(ks, vs))
    ks_s = np.array([x[0] for x in arr])
    vs_s = np.array([x[1] for x in arr])
    return float(np.interp(strike_dec, ks_s, vs_s))


def get_flat_vol(
    vol_surface:    pd.DataFrame,
    cap_maturity_y: float,
    strike_dec:     float,
) -> float:
    """
    Bilinear interpolation (tenor × absolute strike) of the vol surface.

    cap_maturity_y
        TTM to the bond's final payment date (ACT/365, years from spot).
        ONE vol is looked up here and used uniformly for ALL caplets in the
        strip — the market flat-vol convention, analogous to a bond's
        yield-to-maturity applied to an entire strip.

    strike_dec
        Absolute EURIBOR strike in decimal. We interpolate on absolute strike,
        not moneyness, because we supply our own 3M forward rates from the
        discount curve rather than using the surface's 6M ATM rates.

    Tenor-mismatch correction (surface footnote):
        Tenors ≤ 2Y:   surface already quoted on 3M EURIBOR — no scaling.
        Tenors > 2.5Y: surface quoted on 6M EURIBOR — multiply by sqrt(3/6).
    """
    tenors = vol_surface["tenor_years"].values
    if cap_maturity_y <= tenors[0]:
        idx_lo = idx_hi = 0
        w_hi = 0.0
    elif cap_maturity_y >= tenors[-1]:
        idx_lo = idx_hi = len(tenors) - 1
        w_hi = 0.0
    else:
        idx_hi = int(np.searchsorted(tenors, cap_maturity_y, side="right"))
        idx_lo = idx_hi - 1
        w_hi   = ((cap_maturity_y - tenors[idx_lo])
                  / (tenors[idx_hi] - tenors[idx_lo]))

    v_lo = _interp_strike(vol_surface.iloc[idx_lo], strike_dec)
    v_hi = _interp_strike(vol_surface.iloc[idx_hi], strike_dec)
    vol  = (1.0 - w_hi) * v_lo + w_hi * v_hi

    if cap_maturity_y > SCALE_CUTOFF_Y:
        vol *= TENOR_SCALE_3M
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Displaced Black pricer
# ─────────────────────────────────────────────────────────────────────────────

def displaced_black(
    F:      float,
    K:      float,
    T:      float,
    sigma:  float,
    df_pay: float,
    alpha:  float,
    delta:  float = SHIFT,
    phi:    int   = 1,
) -> float:
    """
    Displaced Black caplet / floorlet price per unit notional.

    Formula:
        phi * alpha * P(t, T_pay) * [(F+δ)·N(phi·d1) − (K+δ)·N(phi·d2)]

        d1 = [ln((F+δ)/(K+δ)) + ½σ²T] / (σ√T)
        d2 = d1 − σ√T

    Parameters
    ----------
    F       : 3M forward rate, ACT/360 convention, decimal
    K       : effective EURIBOR strike, decimal
    T       : time to reset date, ACT/365 (years) — the option expiry
    sigma   : flat Displaced Black vol, decimal, already 3M-adjusted
    df_pay  : discount factor to the payment date
    alpha   : caplet accrual factor, ACT/360, decimal
    delta   : displacement parameter δ (default 0.03 = 3%)
    phi     : +1 for caplet, −1 for floorlet

    Returns
    -------
    float : price per unit notional on EURIBOR
    """
    F_s, K_s = F + delta, K + delta
    # Guard: fall back to discounted intrinsic value when vol/time is
    # negligible or shifted rates are non-positive (should not arise with δ=3%)
    if sigma <= 1e-10 or T <= 1e-10 or F_s <= 0.0 or K_s <= 0.0:
        return max(phi * (F - K), 0.0) * alpha * df_pay
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F_s / K_s) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return phi * alpha * df_pay * (F_s * norm.cdf(phi * d1) - K_s * norm.cdf(phi * d2))


# ─────────────────────────────────────────────────────────────────────────────
# Core risk-free bond pricer
# ─────────────────────────────────────────────────────────────────────────────

def price_bond(
    df_fn:               Callable[[date], float],
    holidays:            set,
    schedule:            list,
    spot_date:           date,
    sigma_cap:           float,
    sigma_flr:           float,
    notional:            float = NOTIONAL,
    participation:       float = PARTICIPATION,
    k_eff_cap:           float = K_EFF_CAP,
    k_eff_floor:         float = K_EFF_FLOOR,
    current_coupon_rate: float = CURRENT_COUPON_RATE,
    maturity_date:       date  = MATURITY_DATE,
) -> dict:
    """
    Risk-free structured FRN gross price given a discount-factor closure.

    This is the central repricing engine used across all pricing and
    sensitivity notebooks.

        GP_RF = PV(FRN coupons) + PV(floor strip) − PV(cap strip) + PV(notional)

    Coupon decomposition:
        c(T_i) = 1.6·L_3M + 1.6·max(0% − L_3M, 0) − 1.6·max(L_3M − 3.4063%, 0)

    Parameters
    ----------
    df_fn : callable(date) -> float
        Discount-factor closure from fi_curve.make_df_fn().
        Pass the base-curve closure for the base case; pass a shifted-curve
        closure for DV01 / PCA sensitivity calculations.
        Vols (sigma_cap, sigma_flr) are held fixed across curve shifts —
        they are market inputs independent of the IR curve level.

    sigma_cap, sigma_flr : float
        Single flat vols for the cap and floor strip, from get_flat_vol()
        evaluated once at the bond's full maturity.

    Returns
    -------
    dict with keys:
        gross_price  : RF gross price (EUR per EUR 1000 nominal)
        pv_frn       : PV of the 1.6 × L_3M floating coupons
        pv_cap       : PV of the short cap strip  (positive = cost to bondholder)
        pv_floor     : PV of the long floor strip (positive = benefit to bondholder)
        pv_notional  : PV of the EUR 1000 notional repayment
    """
    pv_frn = pv_cap = pv_floor = 0.0

    for p in schedule:
        reset_date, start_date = p["Reset Date"], p["Start Date"]
        end_date,   pay_date   = p["End Date"],   p["Payment Date"]

        if pay_date <= spot_date:
            continue

        df_end       = df_fn(pay_date)
        coupon_alpha = days_30_360(start_date, end_date) / 360.0   # bond 30/360

        if reset_date < spot_date:
            # First coupon: rate already fixed — use known fixing, no optionality
            pv_frn += notional * current_coupon_rate * coupon_alpha * df_end
        else:
            df_start  = df_fn(start_date)
            fwd_alpha = yearfrac_act_360(start_date, end_date)    # ACT/360 for EURIBOR
            F         = (df_start / df_end - 1.0) / fwd_alpha
            T_reset   = yearfrac_act_365(spot_date, start_date)   # ACT/365 for option

            pv_frn += notional * participation * F * coupon_alpha * df_end

            pv_cap   += participation * notional * displaced_black(
                F, k_eff_cap,   T_reset, sigma_cap, df_end, fwd_alpha, phi=+1)
            pv_floor += participation * notional * displaced_black(
                F, k_eff_floor, T_reset, sigma_flr, df_end, fwd_alpha, phi=-1)

    pv_notional = notional * df_fn(modified_following(maturity_date, holidays))
    gross_price = pv_frn + pv_floor - pv_cap + pv_notional

    return {
        "gross_price": gross_price,
        "pv_frn":      pv_frn,
        "pv_cap":      pv_cap,
        "pv_floor":    pv_floor,
        "pv_notional": pv_notional,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Accrued interest
# ─────────────────────────────────────────────────────────────────────────────

def accrued_interest(
    trade_date:   date,
    period_start: date,
    coupon_rate:  float,
    notional:     float = NOTIONAL,
) -> float:
    """
    Accrued interest on the current fixed coupon, 30/360 basis.
    Dirty price = gross price + accrued interest (EUR bond convention).
    Clean price = gross price − accrued interest.
    """
    return notional * coupon_rate * days_30_360(period_start, trade_date) / 360.0
