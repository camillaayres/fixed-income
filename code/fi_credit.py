"""
fi_credit.py
============
CVA, risky bond pricing, and credit DV01 for the
UniCredit Variable Rate Bond 2034 (ISIN IT0005599110).
Depends on: fi_calendar, fi_bond (which in turn depends on fi_curve)

Exports
-------
survival_prob(cds_bp, T, recovery) -> float
price_risky_bond(df_fn, schedule, spot_date, holidays, cds_bp,
                 sigma_cap, sigma_flr, ...) -> dict
bond_credit_dv01(df_fn, schedule, spot_date, holidays, cds_bp,
                 sigma_cap, sigma_flr, ...) -> float

CVA methodology
---------------
Flat-hazard survival probability:

    Q(0, T) = exp(−h·T),   h = CDS_bp / (10000 · (1 − R))

CVA accumulated over all remaining cashflows:

    CVA = (1 − R) · Σ_i  P(t, T_i) · CF_i · [Q(T_{i-1}) − Q(T_i)]

Risky gross price:

    V_D = V_RF − CVA

Credit DV01 by central finite difference:

    ∂V_D/∂CDS ≈ [V_D(CDS + h) − V_D(CDS − h)] / (2h)
"""

from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np

from fi_calendar import (
    modified_following,
    days_30_360,
    yearfrac_act_360,
    yearfrac_act_365,
)
from fi_bond import (
    price_bond,
    NOTIONAL,
    PARTICIPATION,
    CURRENT_COUPON_RATE,
    MATURITY_DATE,
    K_EFF_CAP,
    K_EFF_FLOOR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Survival probability
# ─────────────────────────────────────────────────────────────────────────────

def survival_prob(cds_bp: float, T: float, recovery: float = 0.40) -> float:
    """
    Flat-hazard survival probability  Q(0, T) = exp(−h · T).

    Parameters
    ----------
    cds_bp   : CDS spread in basis points
    T        : time horizon in years
    recovery : recovery rate (default 0.40)
    """
    h = (cds_bp / 10_000.0) / (1.0 - recovery)
    return float(np.exp(-h * T))


# ─────────────────────────────────────────────────────────────────────────────
# Risky bond pricer
# ─────────────────────────────────────────────────────────────────────────────

def price_risky_bond(
    df_fn:               Callable[[date], float],
    schedule:            list,
    spot_date:           date,
    holidays:            set,
    cds_bp:              float,
    sigma_cap:           float,
    sigma_flr:           float,
    recovery:            float = 0.40,
    notional:            float = NOTIONAL,
    participation:       float = PARTICIPATION,
    current_coupon_rate: float = CURRENT_COUPON_RATE,
    maturity_date:       date  = MATURITY_DATE,
) -> dict:
    """
    Risky structured bond gross price  =  RF gross price  −  CVA.

    Parameters
    ----------
    df_fn     : risk-free DF closure from fi_curve.make_df_fn()
    cds_bp    : issuer 5Y EUR CDS spread in bp (UniCredit: UNIC5YEUAM=R)
    sigma_cap : flat cap vol from fi_bond.get_flat_vol()
    sigma_flr : flat floor vol from fi_bond.get_flat_vol()
    recovery  : recovery rate (default 0.40)

    Returns
    -------
    dict — all keys from fi_bond.price_bond, plus:
        cva         : Credit Valuation Adjustment (EUR per EUR 1000 nominal)
        risky_gross : gross_price − cva  (EUR)
    """
    # Step 1: risk-free price
    rf_result = price_bond(
        df_fn, holidays, schedule, spot_date,
        sigma_cap, sigma_flr, notional, participation,
        K_EFF_CAP, K_EFF_FLOOR, current_coupon_rate, maturity_date,
    )
    rf_gross = rf_result["gross_price"]

    # Step 2: build (T_i, CF_i, DF_i) cashflow list
    cashflows: list[tuple[float, float, float]] = []

    for p in schedule:
        pay_date   = p["Payment Date"]
        start_date = p["Start Date"]
        end_date   = p["End Date"]
        reset_date = p["Reset Date"]

        if pay_date <= spot_date:
            continue

        T_i   = yearfrac_act_365(spot_date, pay_date)
        df_i  = df_fn(pay_date)
        alpha = days_30_360(start_date, end_date) / 360.0

        if reset_date < spot_date:
            cf = notional * current_coupon_rate * alpha
        else:
            df_s = df_fn(start_date)
            fwa  = yearfrac_act_360(start_date, end_date)
            F    = (df_s / df_i - 1.0) / fwa
            cf   = notional * participation * F * alpha

        cashflows.append((T_i, cf, df_i))

    # Add notional repayment at maturity
    mat_adj = modified_following(maturity_date, holidays)
    T_mat   = yearfrac_act_365(spot_date, mat_adj)
    cashflows.append((T_mat, notional, df_fn(mat_adj)))
    cashflows.sort(key=lambda x: x[0])

    # Step 3: accumulate CVA
    #
    # Risky bond valuation formula:
    #   V_D = sum_i Q_i*P_i*coupon_i + Q_n*P_n*N + R*N * sum_i P_i*(Q_{i-1}-Q_i)
    #
    # Therefore:
    #   CVA = V_RF - V_D
    #       = (1-R)*N * sum_i P_i*(Q_{i-1}-Q_i)   [dominant notional default loss]
    #       + sum_i coupon_i*P_i*(1-Q_i)            [coupon default loss, smaller]
    #
    # The notional N appears in EVERY period, not only at maturity. Using cf_i
    # (the period cash flow) instead of N would underestimate CVA because
    # recovery is paid on the notional, not on the scheduled coupon.
    cva, Q_prev = 0.0, 1.0
    for T_i, cf_i, df_i in cashflows:
        Q_i   = survival_prob(cds_bp, T_i, recovery)
        cva  += (1.0 - recovery) * df_i * notional * (Q_prev - Q_i)
        Q_prev = Q_i

    return {
        **rf_result,
        "cva":         cva,
        "risky_gross": rf_gross - cva,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Credit DV01
# ─────────────────────────────────────────────────────────────────────────────

def bond_credit_dv01(
    df_fn:               Callable[[date], float],
    schedule:            list,
    spot_date:           date,
    holidays:            set,
    cds_bp:              float,
    sigma_cap:           float,
    sigma_flr:           float,
    recovery:            float = 0.40,
    bump_bp:             float = 1.0,
    notional:            float = NOTIONAL,
    participation:       float = PARTICIPATION,
    current_coupon_rate: float = CURRENT_COUPON_RATE,
    maturity_date:       date  = MATURITY_DATE,
) -> float:
    """
    Central finite-difference credit DV01 of the risky bond.

    Returns EUR change per +1 bp CDS widening (negative value:
    bond price falls when the issuer's credit spread widens).

    Central finite difference: [V_D(CDS + h) − V_D(CDS − h)] / (2h)

    Parameters
    ----------
    bump_bp : size of the finite-difference bump in basis points (default 1 bp)
    """
    kw = dict(
        df_fn=df_fn, schedule=schedule, spot_date=spot_date, holidays=holidays,
        sigma_cap=sigma_cap, sigma_flr=sigma_flr, recovery=recovery,
        notional=notional, participation=participation,
        current_coupon_rate=current_coupon_rate, maturity_date=maturity_date,
    )
    up = price_risky_bond(cds_bp=cds_bp + bump_bp, **kw)
    dn = price_risky_bond(cds_bp=cds_bp - bump_bp, **kw)
    return (up["risky_gross"] - dn["risky_gross"]) / (2.0 * bump_bp)
