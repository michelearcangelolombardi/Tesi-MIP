# Prezzo del MIP all'ingresso e quota V8.
from __future__ import annotations

from scipy.optimize import brentq

from .bs import call, digitale
from .input import Caso, carica_casi


def fair_value_ingresso(caso: Caso, investimento: float) -> float:
    E, T, r, s, p = caso.equity_ingresso, caso.t_exit, caso.rf_cont_ingresso, caso.vol_ingresso, caso.parametri
    if caso.codice == "A":
        H, R = p["hurdle_mom"] * investimento, p["ratchet_mom"] * investimento
        salto = p["q_ratchet"] - p["q_base"]
        return p["q_base"] * call(E, H, T, r, s) + salto * (call(E, R, T, r, s) + (R - H) * digitale(E, R, T, r, s))
    H = investimento * max(p["hurdle_mom"], (1 + p["hurdle_irr"]) ** T)
    if caso.codice == "B":
        fine = (p["c"] * H - p["s"] * investimento) / (p["c"] - p["s"])
        return p["c"] * call(E, H, T, r, s) - (p["c"] - p["s"]) * call(E, fine, T, r, s)
    return p["p"] * call(E, H, T, r, s)


def prezzo_punto_fisso(caso: Caso) -> float:
    E = caso.equity_ingresso
    return brentq(lambda P: fair_value_ingresso(caso, E - P) - P, 1e-6, 0.5 * E)


def quota_unica_V8(caso: Caso) -> float:
    inv = caso.investimento_fondo
    H = inv * max(caso.parametri["hurdle_mom"], (1 + caso.parametri["hurdle_irr"]) ** caso.t_exit)
    c = call(caso.equity_ingresso, H, caso.t_exit, caso.rf_cont_ingresso, caso.vol_ingresso)
    return fair_value_ingresso(caso, inv) / c


if __name__ == "__main__":
    casi = carica_casi()
    for k, c in casi.items():
        P = prezzo_punto_fisso(c)
        print(f"Caso {k}: prezzo calcolato {P:.4f} €m (arrotondato {P:.2f}) | nel file {c.importo_mip:.2f} | "
              f"quota sull'equity {P / c.equity_ingresso:.4%}")
    print(f"V8, quota unica del Caso B: {quota_unica_V8(casi['B']):.4%}")
