# Passaggio 8: tabelle.
from __future__ import annotations

import pandas as pd

from .alberi import valore_albero, valore_opm_exit_incerta
from .input import Caso, quota_V8
from .montecarlo import valore_mc
from .opm import mercato, valore_opm
from .passaggio7 import tabella_fonti, tabella_gruppi, valori_modelli
from .pwerm import numero_quantili, pwerm_quantili
from .sensitivity import CONV, FINESTRA, INFO, MC, OPM, PRASSI, QUANT, prove, valori
from .waterfall import VARIANTI

PARAMETRI_NUMERICI = {"passi": (25, 50, 100, 200, 500), "simulazioni": (10_000, 50_000, 100_000, 500_000), "quantili": (3, 5, 10, 20, 50)}


def tabella_sensitivity(casi: dict[str, Caso]) -> pd.DataFrame:
    n = numero_quantili()
    righe = []
    for k, c in casi.items():
        base_v = {OPM: valore_opm(c, mercato(c)), MC: valore_mc(c, exit_incerta=True, con_leaver=True).valore}
        base_v.update(valori(prove(c)[0].__class__("", "", 0, c, (PRASSI, QUANT)), "valutazione", n))
        for pr in prove(c):
            vv = valori(pr, "valutazione", n)
            vi = valori(pr, "ingresso", n)
            for m, x in vv.items():
                fi = vi.get(m)
                info = INFO[pr.parametro]
                righe.append({"Caso": k, "Parametro": pr.parametro, "Tipo": info.tipo, "Livello": pr.livello, "Numero": pr.numero,
                              "Nell'intervallo plausibile": "Sì" if (pr.base or info.plausibile(pr.numero)) else "No",
                              "Caso base": "Sì" if pr.base else "",
                              "Modello": m, "Volatilità – valutazione": pr.caso.vol_valutazione,
                              "Valore alla valutazione (€m)": x, "Δ% dal caso base": x / base_v[m] - 1,
                              "Fair value all'ingresso (€m)": fi, "Fair value / equity all'ingresso": fi / pr.caso.equity_ingresso,
                              "Soglia 1% rispettata": "Sì" if fi / pr.caso.equity_ingresso >= 0.01 else "No",
                              "Fair value / (investimento totale − shareholder loan) all'ingresso":
                                  fi / (pr.caso.equity_ingresso - sum(s.importo for s in pr.caso.strumenti_fondo
                                                                      if s.nome.startswith("Shareholder loan"))),
                              "Prezzo pagato (€m)": c.importo_mip})
    return pd.DataFrame(righe)


def tabella_tornado(sens: pd.DataFrame, solo_plausibile: bool = False) -> pd.DataFrame:
    if solo_plausibile:
        sens = sens[sens["Nell'intervallo plausibile"] == "Sì"]
    righe = []
    for (k, m, par), g in sens.groupby(["Caso", "Modello", "Parametro"], sort=False):
        lo, hi = g.loc[g["Δ% dal caso base"].idxmin()], g.loc[g["Δ% dal caso base"].idxmax()]
        info = INFO[par]
        unita = _effetto_unita(g, info)
        righe.append({"Caso": k, "Modello": m, "Parametro": par, "Tipo": info.tipo,
                      "Intervallo": info.motivo if solo_plausibile else "griglia di stress del foglio Sensitivity",
                      "Effetto minimo": lo["Δ% dal caso base"], "Livello minimo": lo.Livello,
                      "Effetto massimo": hi["Δ% dal caso base"], "Livello massimo": hi.Livello,
                      "Ampiezza (max − min)": hi["Δ% dal caso base"] - lo["Δ% dal caso base"],
                      "Scostamento massimo dal base": max(abs(lo["Δ% dal caso base"]), abs(hi["Δ% dal caso base"])),
                      "Effetto per unità": unita[1], "Unità": unita[0]})
    t = pd.DataFrame(righe)
    return t.sort_values(["Caso", "Modello", "Ampiezza (max − min)"], ascending=[True, True, False], kind="stable")


def _effetto_unita(g: pd.DataFrame, info) -> tuple[str, float]:
    if info.unita is None:
        return ("", float("nan"))
    nome, u, a, b = info.unita
    x = g.set_index("Numero")["Δ% dal caso base"]
    if a is None:
        ok = [v for v in x.index if info.plausibile(v)]
        a, b = min(ok), max(ok)
    trova = lambda v: x[[i for i in x.index if abs(i - v) < 1e-9][0]] if any(abs(i - v) < 1e-9 for i in x.index) else None
    xa, xb = trova(a), trova(b)
    if xa is None or xb is None:
        return (nome, float("nan"))
    return (nome, (xb - xa) / ((b - a) / u))


def tabella_classifica(casi: dict[str, Caso], tornado_plaus: pd.DataFrame) -> pd.DataFrame:
    fonti = tabella_fonti(casi)
    righe = []
    for k in casi:
        for _, r in fonti[fonti.Caso == k].iterrows():
            righe.append({"Caso": k, "Tipo": "Modello", "Voce": r["Scelta di modello"], "Scostamento dal caso base": abs(r["Effetto sul valore"]),
                          "Dettaglio": f"{r['Effetto sul valore']:+.1%} ({r['Confronto']})"})
        t = tornado_plaus[(tornado_plaus.Caso == k) & (tornado_plaus.Tipo == "stima")]
        con_mc = set(t[t.Modello == MC].Parametro)
        t = t[(t.Modello == MC) | ((t.Modello == PRASSI) & ~t.Parametro.isin(con_mc))]
        for _, r in t.iterrows():
            righe.append({"Caso": k, "Tipo": "Parametro di stima", "Voce": f"{r.Parametro} [{r.Modello}]",
                          "Scostamento dal caso base": r["Scostamento massimo dal base"],
                          "Dettaglio": f"da {r['Effetto minimo']:+.1%} ({r['Livello minimo']}) a {r['Effetto massimo']:+.1%} ({r['Livello massimo']})"})
    df = pd.DataFrame(righe).sort_values(["Caso", "Scostamento dal caso base"], ascending=[True, False], kind="stable")
    df.insert(1, "Rango", df.groupby("Caso").cumcount() + 1)
    return df


def tabella_model_vs_parameter(casi: dict[str, Caso], tornado_plaus: pd.DataFrame) -> pd.DataFrame:
    gruppi = tabella_gruppi({k: valori_modelli(c, "valutazione") for k, c in casi.items()})
    righe = []
    for k in casi:
        g = gruppi[gruppi.Caso == k].set_index(gruppi[gruppi.Caso == k]["Gruppo di modelli"].str[:2])
        t = tornado_plaus[(tornado_plaus.Caso == k) & (tornado_plaus.Modello == MC) & (tornado_plaus.Tipo == "stima")]
        t = t.sort_values("Ampiezza (max − min)", ascending=False)
        g2, g3 = g.loc["2.", "Dispersione"], g.loc["3.", "Dispersione"]
        righe.append({"Caso": k,
                      "Model risk: OPM semplice vs modelli completi (gruppo 2)": g2,
                      "Model risk: + strutturale (gruppo 3)": g3,
                      "Parametro di stima più influente": t.iloc[0].Parametro,
                      "Sua ampiezza (intervallo plausibile)": t.iloc[0]["Ampiezza (max − min)"],
                      "Parametri di stima con ampiezza > gruppo 2": int((t["Ampiezza (max − min)"] > g2).sum()),
                      "Parametri di stima con ampiezza > gruppo 3": int((t["Ampiezza (max − min)"] > g3).sum()),
                      "Parametri di stima valutati": len(t)})
    return pd.DataFrame(righe)


def tabella_art60(sens: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for (k, m), g in sens[sens.Modello.isin([OPM, MC])].groupby(["Caso", "Modello"]):
        riga_base = g[g["Caso base"] == "Sì"].iloc[0]
        base = riga_base["Fair value / equity all'ingresso"]
        g = g[(g.Parametro != "Definizione di leva") & (g["Caso base"] != "Sì")]
        pl = g[(g.Tipo == "stima") & (g["Nell'intervallo plausibile"] == "Sì")]
        st = g[g.Tipo != "convenzione"]
        cv = g[g.Parametro == CONV]
        for nome, x in (("Parametri di stima, intervallo plausibile", pl), ("Tutti i parametri, griglia di stress", st),
                        ("Convenzione di volatilità", cv)):
            minimo = x.loc[x["Fair value / equity all'ingresso"].idxmin()]
            sotto = x[x["Soglia 1% rispettata"] == "No"]
            righe.append({"Caso": k, "Modello": m, "Prove": nome, "Caso base": base,
                          "Minimo": minimo["Fair value / equity all'ingresso"], "Prova del minimo": f"{minimo.Parametro}: {minimo.Livello}",
                          "Prove sotto l'1%": len(sotto), "Prove valutate": len(x),
                          "Quali prove sono sotto l'1%": "; ".join(f"{r.Parametro}: {r.Livello}" for _, r in sotto.iterrows()),
                          "Denominatore alternativo (senza shareholder loan), caso base":
                              riga_base["Fair value / (investimento totale − shareholder loan) all'ingresso"]})
    return pd.DataFrame(righe)


def tabella_convenzione(sens: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for (k, m), g in sens[(sens.Parametro == CONV)].groupby(["Caso", "Modello"], sort=False):
        b, p = g[g["Caso base"] == "Sì"].iloc[0], g[g["Caso base"] != "Sì"].iloc[0]
        righe.append({"Caso": k, "Modello": m,
                      "Volatilità relevered – valutazione": b["Volatilità – valutazione"],
                      "Volatilità dei peer senza relevering – valutazione": p["Volatilità – valutazione"],
                      "Valore relevered (€m)": b["Valore alla valutazione (€m)"],
                      "Valore senza relevering (€m)": p["Valore alla valutazione (€m)"],
                      "Δ% dal caso base": p["Δ% dal caso base"],
                      "Fair value / equity all'ingresso – relevered": b["Fair value / equity all'ingresso"],
                      "Fair value / equity all'ingresso – senza relevering": p["Fair value / equity all'ingresso"],
                      "Soglia 1% rispettata – senza relevering": p["Soglia 1% rispettata"]})
    return pd.DataFrame(righe)


def tabella_numerici(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        bs = valore_opm(c, mercato(c))
        for nome, valori_ in (("Passi dell'albero binomiale", [valore_albero(c, tipo="binomiale", N=n) for n in PARAMETRI_NUMERICI["passi"]]),
                              ("Simulazioni Monte Carlo (senza controllo)", [valore_mc(c, n=n, controllo=False).valore for n in PARAMETRI_NUMERICI["simulazioni"]]),
                              ("Quantili del PWERM", [pwerm_quantili(c, n=n)[0] for n in PARAMETRI_NUMERICI["quantili"]])):
            rif = bs if not nome.startswith("Quantili") else pwerm_quantili(c, n=1000)[0]
            righe.append({"Caso": k, "Impostazione": nome, "Minimo (€m)": min(valori_), "Massimo (€m)": max(valori_),
                          "Riferimento esatto (€m)": rif, "Ampiezza / riferimento": (max(valori_) - min(valori_)) / rif})
    return pd.DataFrame(righe)


def tabella_varianti(casi: dict[str, Caso]) -> pd.DataFrame:
    q8 = quota_V8()
    righe = []
    for k, c in casi.items():
        base = None
        for v in VARIANTI[k]:
            q = q8 if v == "V8" else None
            x = {"OPM, exit fissa": valore_opm(c, mercato(c), variante=v, quota_unica=q),
                 "OPM, exit incerta": valore_opm_exit_incerta(c, variante=v, quota_unica=q),
                 "Monte Carlo completo": valore_mc(c, exit_incerta=True, con_leaver=True, variante=v, quota_unica=q).valore}
            base = base or x
            righe.append({"Caso": k, "Variante": v or "Caso base", **{f"{m} (€m)": val for m, val in x.items()},
                          **{f"Δ% {m}": val / base[m] - 1 for m, val in x.items()}})
    return pd.DataFrame(righe)


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    sens = tabella_sensitivity(casi)
    stress = tabella_tornado(sens)
    plaus = tabella_tornado(sens, solo_plausibile=True)
    return {
        "Sensitivity": sens,
        "Tornado_stima_plausibile": plaus[plaus.Tipo == "stima"],
        "Tornado_contratto_struttura": stress[stress.Tipo.isin(["contratto", "struttura"])],
        "Tornado_stress_completo": stress,
        "Classifica_modello_parametri": tabella_classifica(casi, plaus),
        "Model_vs_parameter_risk": tabella_model_vs_parameter(casi, plaus),
        "Art60_sensitivity": tabella_art60(sens),
        "Convenzione_volatilita": tabella_convenzione(sens),
        "Impostazioni_numeriche": tabella_numerici(casi),
        "Varianti_V1_V8": tabella_varianti(casi),
    }
