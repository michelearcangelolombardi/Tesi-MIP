# Ipotesi H1 sullo stesso deal.
import statistics

import pandas as pd

from mip.alberi import valore_albero
from mip.config import CARTELLA_RISULTATI
from mip.input import carica_casi
from mip.montecarlo import valore_mc
from mip.opm import mercato, valore_opm
from mip.strutturale import sigma_attivo_coerente

VARIANTI = [
    ("A", None, "Ratchet con salto (base)"),
    ("A", "V5", "Ratchet smussato su una rampa di ±5%"),
    ("A", "V6", "Ratchet incrementale, senza salto"),
    ("B", None, "Catch-up (base)"),
    ("B", "V7", "Senza catch-up, quota s lineare"),
]


def dispersione(v):
    return (max(v) - min(v)) / statistics.median(v)


def valori(caso, variante, data="valutazione"):
    full = dict(exit_incerta=True, con_leaver=True, variante=variante)
    return {
        "OPM semplice": valore_opm(caso, mercato(caso, data), variante=variante),
        "Albero binomiale completo": valore_albero(caso, data, "binomiale", **full),
        "Albero trinomiale completo": valore_albero(caso, data, "trinomiale", **full),
        "Monte Carlo completo": valore_mc(caso, data, **full).valore,
        "Strutturale (σ attivo dei peer)": valore_mc(caso, data, modello="strutturale", **full).valore,
        "Strutturale (σ attivo coerente)": valore_mc(caso, data, modello="strutturale",
                                                     sigma_attivo=sigma_attivo_coerente(caso, data), **full).valore,
    }


def main():
    casi = carica_casi()
    righe = []
    for k, var, descr in VARIANTI:
        v = valori(casi[k], var)
        g1 = [v["Albero binomiale completo"], v["Albero trinomiale completo"], v["Monte Carlo completo"]]
        g2 = g1 + [v["OPM semplice"]]
        righe.append({"Caso": k, "Variante": var or "base", "Payoff": descr, **{m: float(x) for m, x in v.items()},
                      "Dispersione gruppo 1": dispersione(g1), "Dispersione gruppo 2": dispersione(g2),
                      "Dispersione gruppo 3": dispersione(list(v.values()))})
    df = pd.DataFrame(righe)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df[["Caso", "Variante", "Payoff", "Dispersione gruppo 1", "Dispersione gruppo 2", "Dispersione gruppo 3"]].to_string(index=False))
    df.to_excel(CARTELLA_RISULTATI / "11_stesso_deal.xlsx", index=False)


if __name__ == "__main__":
    main()
