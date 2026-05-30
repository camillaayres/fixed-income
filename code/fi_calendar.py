"""
fi_calendar.py
==============
Business-day calendar and day-count conventions for the
UniCredit Variable Rate Bond 2034 (ISIN IT0005599110).
No dependencies on other fi_ modules.

Exports
-------
load_target_holidays(path) -> set[date]
is_business_day(d, holidays) -> bool
modified_following(d, holidays) -> date
add_business_days(d, n, holidays) -> date
subtract_business_days(d, n, holidays) -> date
days_30_360(d1, d2) -> int
yearfrac_act_360(d1, d2) -> float
yearfrac_act_365(d1, d2) -> float
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Holiday calendar
# ─────────────────────────────────────────────────────────────────────────────

def load_target_holidays(path: str) -> set:
    """
    Parse TARGET holiday dates from Holidays.xlsx.
    Expects one date per row in the first column; string headers are skipped.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None)
    out: set = set()
    for x in raw.iloc[:, 0].dropna():
        if isinstance(x, str):
            continue
        if isinstance(x, pd.Timestamp):
            out.add(x.date())
        elif isinstance(x, (datetime, date)):
            out.add(x if isinstance(x, date) else x.date())
    return out


def is_business_day(d: date, holidays: set) -> bool:
    return d.weekday() < 5 and d not in holidays


def modified_following(d: date, holidays: set) -> date:
    """
    Advance to the next business day; if that crosses a month boundary,
    go back to the last business day in the original month.
    """
    orig_month, dd = d.month, d
    while not is_business_day(dd, holidays):
        dd += timedelta(days=1)
    if dd.month != orig_month:
        dd -= timedelta(days=1)
        while not is_business_day(dd, holidays):
            dd -= timedelta(days=1)
    return dd


def add_business_days(d: date, n: int, holidays: set) -> date:
    dd, step, count = d, (1 if n >= 0 else -1), 0
    while count < abs(n):
        dd += timedelta(days=step)
        if is_business_day(dd, holidays):
            count += 1
    return dd


def subtract_business_days(d: date, n: int, holidays: set) -> date:
    return add_business_days(d, -n, holidays)


# ─────────────────────────────────────────────────────────────────────────────
# Day-count conventions
# ─────────────────────────────────────────────────────────────────────────────

def days_30_360(d1: date, d2: date) -> int:
    """
    30/360 Bond Basis — coupon cash-flow accrual (Final Terms convention).
    """
    y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if dd2 == 31 and dd1 >= 30:
        dd2 = 30
    return (y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)


def yearfrac_act_360(d1: date, d2: date) -> float:
    """ACT/360 — EURIBOR forward-rate extraction and caplet accrual."""
    return (d2 - d1).days / 360.0


def yearfrac_act_365(d1: date, d2: date) -> float:
    """ACT/365 — option time to expiry (T in d1/d2 formula)."""
    return (d2 - d1).days / 365.0
