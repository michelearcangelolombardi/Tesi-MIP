# Passaggio 6: tabelle.
from __future__ import annotations

import pandas as pd

from .alberi import distribuzione_exit
from .input import Caso, anni
from .lsm import parametri_lsm, valuta
from .montecarlo import valore_mc


def _t_medio_distribuzione(caso: Caso, data: str) -> float:
    rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    return sum(p * anni(rif, d) for d, p in distribuzione_exit(caso) if d > rif)


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    valori, distr, fondo, conflitto = [], [], [], []
    for data in ("valutazione", "ingresso"):
        for k, c in casi.items():
            if not parametri_lsm(c).applicare:
                continue
            E0 = c.equity_valutazione if data == "valutazione" else c.equity_ingresso
            res = valuta(c, data)
            fissa = valore_mc(c, data)
            incerta = valore_mc(c, data, exit_incerta=True)
            t_dist = _t_medio_distribuzione(c, data)
            righe = [("Exit fissa (caso base)", fissa.valore, fissa.errore_standard, c.T if data == "valutazione" else c.t_exit, None),
                     ("Exit incerta (Exit_Distribuzione)", incerta.valore, incerta.errore_standard, t_dist, None)]
            for nome in ("Regola obiettivi", "LSM (ottimo del fondo)"):
                e = res[nome]
                reale = res["Regola obiettivi – misura reale" if nome == "Regola obiettivi" else "LSM – misura reale"]
                righe.append((nome, e.valore_mip, e.errore_standard, e.tempo_medio, reale.tempo_medio))
            pf, pa = parametri_lsm(c, finestre_foglio=True), parametri_lsm(c)
            ext = None
            if (pf.prima, pf.ultima) != (pa.prima, pa.ultima):
                ext = valuta(c, data, finestre_foglio=True)
                for nome in ("Regola obiettivi", "LSM (ottimo del fondo)"):
                    e = ext[nome]
                    reale = ext["Regola obiettivi – misura reale" if nome == "Regola obiettivi" else "LSM – misura reale"]
                    righe.append((f"{nome}, finestre {pf.prima}–{pf.ultima}", e.valore_mip, e.errore_standard,
                                  e.tempo_medio, reale.tempo_medio))
            for nome, v, se, tq, tp in righe:
                valori.append({"Data": data, "Caso": k, "Regola di exit": nome, "Valore del MIP (€m)": v, "Errore standard (€m)": se,
                               "Δ% da exit incerta": v / incerta.valore - 1, "Valore / equity": v / E0,
                               "Tempo medio di uscita (anni, misura risk-neutral)": tq,
                               "Tempo medio di uscita (anni, misura reale)": tp,
                               "Fair value − prezzo (€m)": v - c.importo_mip if data == "ingresso" else None})
            if data != "valutazione":
                continue
            fin_a, fin_f = f"{pa.prima}–{pa.ultima}", f"{pf.prima}–{pf.ultima}"
            for insieme, fin in ((res, fin_a), (ext, fin_f)):
                if insieme is None:
                    continue
                for nome in ("Regola obiettivi", "LSM (ottimo del fondo)"):
                    e = insieme[nome]
                    for cf in e.conflitto:
                        conflitto.append({"Caso": k, "Regola": nome, "Finestre": fin, "Politica": e.regola,
                                          "Soglia": {"hurdle": "hurdle", "ratchet": "ratchet", "fine_catch_up": "fine del catch-up"}[cf["soglia"]],
                                          "P(exit | equity tra 90% e 100% della soglia)": cf["p_exit_appena_sotto"],
                                          "P(exit | equity tra 100% e 110% della soglia)": cf["p_exit_appena_sopra"],
                                          "Osservazioni sotto": cf["casi_sotto"], "Osservazioni sopra": cf["casi_sopra"]})
            for nome, e in res.items():
                if nome.startswith("Exit fissa"):
                    fondo.append({"Caso": k, "Regola": nome, "Valore del MIP (€m)": e.valore_mip,
                                  "Proventi netti del fondo, valore attuale (€m)": e.valore_fondo})
                else:
                    distr.append({"Caso": k, "Regola": nome, **{str(y): p for y, p in e.distribuzione.items()},
                                  "Tempo medio (anni)": e.tempo_medio})
            for nome in ("Regola obiettivi", "LSM (ottimo del fondo)"):
                fondo.append({"Caso": k, "Regola": nome,
                              "Valore del MIP (€m)": res[nome].valore_mip,
                              "Proventi netti del fondo, valore attuale (€m)": res[nome].valore_fondo})
            if ext is not None:
                for nome in ("Regola obiettivi", "LSM (ottimo del fondo)"):
                    fondo.append({"Caso": k, "Regola": f"{nome}, finestre {fin_f}", "Valore del MIP (€m)": ext[nome].valore_mip,
                                  "Proventi netti del fondo, valore attuale (€m)": ext[nome].valore_fondo})
            distr.append({"Caso": k, "Regola": "Exit_Distribuzione (foglio Excel)",
                          **{str(d.year): p for d, p in distribuzione_exit(c)}, "Tempo medio (anni)": t_dist})
    distr = pd.DataFrame(distr)
    anni_col = sorted(c for c in distr.columns if c.isdigit())
    distr = distr[["Caso", "Regola"] + anni_col + ["Tempo medio (anni)"]].fillna(0.0)
    return {"Valori": pd.DataFrame(valori), "Distribuzione_exit": distr,
            "Fondo_LSM_vs_fisse": pd.DataFrame(fondo), "Conflitto_soglie": pd.DataFrame(conflitto)}
