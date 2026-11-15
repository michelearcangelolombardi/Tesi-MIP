# Formule di Black-Scholes.
from __future__ import annotations

import math

from scipy.stats import norm


def call(S: float, K: float, T: float, r: float, s: float) -> float:
    if K <= 0:
        return S - K * math.exp(-r * T)
    d1 = (math.log(S / K) + (r + s * s / 2) * T) / (s * math.sqrt(T))
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - s * math.sqrt(T))


def digitale(S: float, K: float, T: float, r: float, s: float) -> float:
    if K <= 0:
        return math.exp(-r * T)
    d2 = (math.log(S / K) + (r - s * s / 2) * T) / (s * math.sqrt(T))
    return math.exp(-r * T) * norm.cdf(d2)
