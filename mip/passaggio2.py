# Passaggio 2: tabelle.
from __future__ import annotations

import pandas as pd

from .input import Caso, quota_V8
from .opm import componente_ratchet, mercato, scomposizione_opm, valore_opm
from .pwerm import limite_quantili, numero_quantili, pwerm_prassi, pwerm_quantili, scomposizione
from .waterfall import VARIANTI

SOGLIA_ART60 = 0.01
QUANTILI_CONVERGENZA = (3, 5, 10, 20, 50, 100, 1000)


def _per_azione(caso: Caso, valore_m: float) -> float:
    return valore_m * 1e6 / caso.n_mip


def tabella_sintesi(casi: dict[str, Caso], n: int) -> pd.DataFrame:
    righe = []
    for data in ("valutazione", "ingresso"):
        for k, c in casi.items():
            E0 = c.equity_valutazione if data == "valutazione" else c.equity_ingresso
            m = mercato(c, data)
            valori = {"OPM (Black-Scholes)": valore_opm(c, m),
                      "PWERM da prassi": pwerm_prassi(c, data)[0],
                      f"PWERM da quantili (N={n})": pwerm_quantili(c, data, n)[0]}
            for modello, v in valori.items():
                righe.append({"Data": data, "Caso": k, "Modello": modello, "Valore del MIP (€m)": v,
                              "Valore per azione (€)": _per_azione(c, v), "Valore / equity": v / E0,
                              "Componente ratchet (€m)": componente_ratchet(c, m) if (k == "A" and modello.startswith("OPM")) else None})
    return pd.DataFrame(righe)


def tabella_art60(casi: dict[str, Caso], n: int) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        valori = {"OPM (Black-Scholes)": valore_opm(c, mercato(c, "ingresso")),
                  "PWERM da prassi": pwerm_prassi(c, "ingresso")[0],
                  f"PWERM da quantili (N={n})": pwerm_quantili(c, "ingresso", n)[0]}
        for modello, v in valori.items():
            quota = v / c.equity_ingresso
            righe.append({"Caso": k, "Modello": modello, "Fair value all'ingresso (€m)": v,
                          "Fair value / equity all'ingresso": quota,
                          "Soglia 1% rispettata": "Sì" if quota >= SOGLIA_ART60 else "No",
                          "Prezzo pagato (€m)": c.importo_mip,
                          "Fair value − prezzo (€m)": v - c.importo_mip})
    return pd.DataFrame(righe)


def tabella_breakpoint(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        v, det = scomposizione_opm(c, mercato(c, "valutazione"))
        for d in det:
            righe.append({"Caso": k, **d})
        righe.append({"Caso": k, "breakpoint": "Totale", "contributo": v})
    return pd.DataFrame(righe)


def tabella_scenari(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for data in ("valutazione", "ingresso"):
        for k, c in casi.items():
            for r in pwerm_prassi(c, data)[1]:
                righe.append({"Data": data, "Caso": k, **r})
    return pd.DataFrame(righe)


def tabella_quantili(casi: dict[str, Caso], n: int) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        for r in pwerm_quantili(c, "valutazione", n)[1]:
            righe.append({"Caso": k, **r})
    return pd.DataFrame(righe)


def tabella_convergenza_quantili(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        riga = {"Caso": k}
        for n in QUANTILI_CONVERGENZA:
            riga[f"N={n}"] = pwerm_quantili(c, "valutazione", n)[0]
        riga["N→∞"] = limite_quantili(c, "valutazione")
        righe.append(riga)
    return pd.DataFrame(righe)


def tabella_scomposizione(casi: dict[str, Caso], n: int) -> pd.DataFrame:
    return pd.DataFrame({k: scomposizione(c, "valutazione", n) for k, c in casi.items()})


def tabella_varianti(casi: dict[str, Caso]) -> pd.DataFrame:
    q8 = quota_V8()
    righe = []
    for k, c in casi.items():
        m = mercato(c, "valutazione")
        base = valore_opm(c, m)
        for v in VARIANTI[k]:
            val = valore_opm(c, m, variante=v, quota_unica=q8 if v == "V8" else None)
            righe.append({"Caso": k, "Variante": v or "Caso base", "Valore OPM (€m)": val,
                          "Differenza dal caso base (€m)": val - base, "Differenza %": val / base - 1})
    return pd.DataFrame(righe)


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    n = numero_quantili()
    return {
        "Sintesi": tabella_sintesi(casi, n),
        "Art60_ingresso": tabella_art60(casi, n),
        "Scomposizione": tabella_scomposizione(casi, n),
        "OPM_breakpoint": tabella_breakpoint(casi),
        "PWERM_scenari": tabella_scenari(casi),
        "PWERM_quantili": tabella_quantili(casi, n),
        "Convergenza_quantili": tabella_convergenza_quantili(casi),
        "Varianti_OPM": tabella_varianti(casi),
    }
