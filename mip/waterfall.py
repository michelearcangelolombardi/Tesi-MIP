# Waterfall dei tre casi.
from __future__ import annotations

import numpy as np

from .input import Caso

VARIANTI = {"A": (None, "V5", "V6"), "B": (None, "V1", "V2", "V7", "V8"), "C": (None, "V3", "V4")}


def _hurdle(caso: Caso, t, variante=None):
    p, inv = caso.parametri, caso.investimento_fondo
    if caso.codice == "A":
        return np.full_like(np.asarray(t, dtype=float), p["hurdle_mom"] * inv)
    if variante in ("V1", "V3"):
        return inv * (1 + p["hurdle_irr"]) ** t
    if variante in ("V2", "V4"):
        return np.full_like(np.asarray(t, dtype=float), inv * (1 + p["hurdle_irr"]) ** caso.t_exit)
    return inv * np.maximum(p["hurdle_mom"], (1 + p["hurdle_irr"]) ** t)


def soglie(caso: Caso, t, variante=None) -> dict:
    t = np.asarray(t, dtype=float)
    H = _hurdle(caso, t, variante)
    if caso.codice == "A":
        return {"hurdle": H, "ratchet": np.full_like(t, caso.parametri["ratchet_mom"] * caso.investimento_fondo)}
    if caso.codice == "B":
        c, s, inv = caso.parametri["c"], caso.parametri["s"], caso.investimento_fondo
        return {"hurdle": H, "fine_catch_up": (c * H - s * inv) / (c - s)}
    return {"hurdle": H}


def payoff(caso: Caso, E, t, variante=None, delta: float = 0.05, quota_unica: float | None = None) -> dict:
    if variante not in VARIANTI[caso.codice]:
        raise ValueError(f"Variante {variante} non prevista per il Caso {caso.codice}")
    E = np.asarray(E, dtype=float)
    t = np.asarray(t, dtype=float)
    E, t = np.broadcast_arrays(E, t)
    p, inv = caso.parametri, caso.investimento_fondo

    shl, pref = caso.strumento("Shareholder loan"), caso.strumento("Strumento privilegiato")
    shl_cap = shl.importo * (1 + shl.pik) ** t if shl else np.zeros_like(E)
    pref_cap = pref.importo * (1 + pref.pik) ** t if pref else np.zeros_like(E)
    rimb_shl = np.minimum(E, shl_cap)
    rimb_pref = np.minimum(E - rimb_shl, pref_cap)
    residuo = E - rimb_shl - rimb_pref
    F = rimb_shl + rimb_pref + residuo

    sg = soglie(caso, t, variante)
    H = sg["hurdle"]
    eccedenza = np.maximum(0.0, F - H)

    if caso.codice == "A":
        q1, q2, L = p["q_base"], p["q_ratchet"], sg["ratchet"]
        if variante == "V5":
            peso = np.clip((F - L * (1 - delta)) / (2 * delta * L), 0.0, 1.0)
            mip = np.where(F >= H, q1 + (q2 - q1) * peso, 0.0) * eccedenza
        elif variante == "V6":
            mip = q1 * eccedenza + (q2 - q1) * np.maximum(0.0, F - L)
        else:
            q = np.where(F >= L, q2, np.where(F >= H, q1, 0.0))
            mip = q * eccedenza
    elif caso.codice == "B":
        c, s = p["c"], p["s"]
        if variante == "V7":
            mip = s * eccedenza
        elif variante == "V8":
            if quota_unica is None:
                raise ValueError("La variante V8 richiede quota_unica")
            mip = quota_unica * eccedenza
        else:
            K = sg["fine_catch_up"]
            mip = c * np.maximum(0.0, np.minimum(F, K) - H) + s * np.maximum(0.0, F - K)
    else:
        mip = p["p"] * eccedenza

    mip = np.minimum(mip, residuo)

    return {
        "mip": mip,
        "fondo_shl": rimb_shl,
        "fondo_pref": rimb_pref,
        "fondo_ordinarie": residuo - mip,
        "proventi_lordi_fondo": F,
        "proventi_netti_fondo": F - mip,
        "eccedenza": eccedenza,
        **{f"soglia_{n}": v for n, v in sg.items()},
    }
