# Esecuzione dei passaggi 1-10.
import datetime as dt
import sys

import pandas as pd

from mip.config import CARTELLA_RISULTATI, FILE_INPUT, TOLL_SOGLIE, TOLL_WATERFALL
from mip import passaggio2, passaggio3, passaggio4, passaggio5, passaggio6, passaggio7, passaggio8, passaggio9, passaggio10
from mip.input import carica_casi, hash_file
from mip.verifica import confronta_test_breakpoint, confronta_waterfall
from mip.waterfall import soglie


def riepilogo_input(casi) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        sg = {n: round(float(v), 2) for n, v in soglie(c, c.t_exit).items()}
        righe.append({
            "Caso": k, "Struttura del MIP": c.struttura,
            "Equity all'ingresso": c.equity_ingresso, "Investimento del fondo": round(c.investimento_fondo, 2),
            "Prezzo del MIP (€m)": c.importo_mip, "Prezzo per azione (€)": c.prezzo_mip, "Prezzo / equity all'ingresso": round(c.importo_mip / c.equity_ingresso, 4),
            "Equity alla valutazione": round(c.equity_valutazione, 2), "T (anni)": round(c.T, 4), "t exit (anni)": round(c.t_exit, 4),
            "Volatilità valutazione": c.vol_valutazione, "Volatilità ingresso": c.vol_ingresso,
            "Risk-free continuo valutazione": round(c.rf_cont_valutazione, 5), "Ke valutazione": round(c.ke_valutazione, 5),
            "Soglie all'exit base": sg, "Parametri del MIP": c.parametri,
        })
    return pd.DataFrame(righe)


def intestazione():
    print("=" * 78)
    print(f"Esecuzione del {dt.datetime.now():%d/%m/%Y %H:%M}")
    print(f"File di input: {FILE_INPUT.name}  (impronta {hash_file(FILE_INPUT)})")
    print("=" * 78)


def foglio_info():
    return pd.DataFrame({"Voce": ["Data esecuzione", "File di input", "Impronta del file"],
                         "Valore": [f"{dt.datetime.now():%d/%m/%Y %H:%M}", FILE_INPUT.name, hash_file(FILE_INPUT)]})


def passaggio_1(casi):
    rip = riepilogo_input(casi)
    print("\nPASSAGGIO 1 – Input letti dal file Excel")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(rip.drop(columns=["Parametri del MIP", "Soglie all'exit base"]).to_string(index=False))
    for k, c in casi.items():
        print(f"  Caso {k}: parametri {c.parametri} | soglie all'exit base {rip.loc[rip.Caso == k, 'Soglie all' + chr(39) + 'exit base'].iloc[0]}")

    print("\nPASSAGGIO 1 – Verifica della waterfall contro i fogli Waterfall_A/B/C")
    wf = pd.DataFrame(confronta_waterfall(casi))
    tb = pd.DataFrame(confronta_test_breakpoint(casi))
    n_col = wf.groupby("caso")["colonna"].nunique()
    print(f"  Colonne di test: A {n_col['A']}, B {n_col['B']}, C {n_col['C']} (totale {n_col.sum()}); confronti: {len(wf)}")
    print(f"  Differenza massima waterfall: {wf.differenza.max():.2e}  (tolleranza {TOLL_WATERFALL:.0e})  -> {'OK' if wf.differenza.max() < TOLL_WATERFALL else 'ERRORE'}")
    print(f"  Differenza massima Test_Breakpoint: {tb.differenza.max():.4f}  (tolleranza {TOLL_SOGLIE})  -> {'OK' if tb.differenza.max() < TOLL_SOGLIE else 'ERRORE'}")

    uscita = CARTELLA_RISULTATI / "01_verifica_waterfall.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        rip.astype({"Soglie all'exit base": str, "Parametri del MIP": str}).to_excel(xw, sheet_name="Input", index=False)
        wf.to_excel(xw, sheet_name="Waterfall", index=False)
        tb.to_excel(xw, sheet_name="Test_Breakpoint", index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def _stampa(titolo, df, formato="{:,.3f}"):
    print(f"\n{titolo}")
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.float_format", formato.format):
        indice_semplice = not isinstance(df.index, pd.MultiIndex) and pd.api.types.is_integer_dtype(df.index)
        print(df.to_string(index=not indice_semplice))


def passaggio_2(casi):
    t = passaggio2.esegui(casi)
    print("\n" + "=" * 78 + "\nPASSAGGIO 2 – OPM e PWERM\n" + "=" * 78)

    s = t["Sintesi"]
    for data in ("valutazione", "ingresso"):
        piv = s[s.Data == data].pivot(index="Modello", columns="Caso", values="Valore del MIP (€m)")
        _stampa(f"Valore del MIP (€m) alla data di {data}", piv)
    a = s[(s.Caso == "A") & s.Modello.str.startswith("OPM")]
    print(f"\n  Caso A, componente ratchet (OPM): valutazione {a.iloc[0]['Componente ratchet (€m)']:.3f} €m, "
          f"ingresso {a.iloc[1]['Componente ratchet (€m)']:.3f} €m")

    _stampa("Scomposizione PWERM vs OPM (€m, data di valutazione)", t["Scomposizione"])

    art = t["Art60_ingresso"].copy()
    art["Fair value / equity all'ingresso"] = art["Fair value / equity all'ingresso"].map("{:.2%}".format)
    _stampa("Test art. 60 all'ingresso", art)

    _stampa("Convergenza del PWERM da quantili (€m, data di valutazione)", t["Convergenza_quantili"])

    sc = t["PWERM_scenari"]
    sc = sc[sc.Data == "valutazione"][["Caso", "scenario", "anno_exit", "probabilita", "equity_exit", "regime", "mip_exit", "valore_attuale_pesato"]]
    _stampa("Scenari di prassi e soglie (data di valutazione)", sc)

    v = t["Varianti_OPM"].copy()
    v["Differenza %"] = v["Differenza %"].map("{:+.1%}".format)
    _stampa("Varianti, valore OPM (data di valutazione)", v)

    uscita = CARTELLA_RISULTATI / "02_opm_pwerm.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome, index=(nome == "Scomposizione"))
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def passaggio_3(casi):
    t = passaggio3.esegui(casi)
    print("\n" + "=" * 78 + "\nPASSAGGIO 3 – Alberi binomiale e trinomiale\n" + "=" * 78)

    v = t["Valori"]
    for data in ("valutazione", "ingresso"):
        piv = v[v.Data == data].pivot_table(index=["Configurazione", "Modello"], columns="Caso",
                                            values="Valore del MIP (€m)", sort=False)
        _stampa(f"Valore del MIP (€m) alla data di {data}", piv)
    a = v[(v.Caso == "A") & (v.Data == "valutazione")][["Configurazione", "Modello", "Componente ratchet (€m)"]].dropna()
    _stampa("Caso A, componente ratchet alla data di valutazione (€m)", a)

    d = t["Dispersione"].copy()
    d["Dispersione"] = d["Dispersione"].map("{:.1%}".format)
    _stampa("Dispersione tra modelli option-based", d[d.Data == "valutazione"].drop(columns="Data"))

    art = t["Art60_ingresso"].copy()
    art["Valore / equity"] = art["Valore / equity"].map("{:.2%}".format)
    _stampa("Test art. 60 all'ingresso", art)

    _stampa("Convergenza degli alberi (€m, exit fissa, data di valutazione)", t["Convergenza_alberi"])

    va = t["Varianti"].copy()
    for col in ("Δ% exit fissa", "Δ% exit incerta"):
        va[col] = va[col].map("{:+.1%}".format)
    _stampa("Varianti: exit fissa ed exit incerta", va)

    uscita = CARTELLA_RISULTATI / "03_alberi.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome, index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def passaggio_4(casi):
    print("\n" + "=" * 78 + "\nPASSAGGIO 4 – Monte Carlo\n" + "=" * 78)
    t = passaggio4.esegui(casi)

    mc = t["Monte_Carlo"]
    for data in ("valutazione", "ingresso"):
        x = mc[mc.Data == data][["Caso", "Configurazione", "Valore del MIP (€m)", "Errore standard (€m)", "IC 95% min", "IC 95% max",
                                 "Senza variabile di controllo (€m)", "Errore standard senza controllo (€m)"]]
        _stampa(f"Monte Carlo alla data di {data} (€m)", x)
    a = mc[(mc.Caso == "A") & (mc.Data == "valutazione")][["Configurazione", "Componente ratchet (€m)"]]
    _stampa("Caso A, componente ratchet con il Monte Carlo (€m, data di valutazione)", a)

    mp = t["Mappa_convergenza"]
    mp = mp[mp.Data == "valutazione"].drop(columns="Data").copy()
    for col in ("Dispersione", "Dispersione componente ratchet"):
        if col in mp:
            mp[col] = mp[col].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    _stampa("Mappa di convergenza tra modelli option-based (€m, data di valutazione)", mp.astype(object).where(mp.notna(), ""))

    pw = t["PWERM_vs_option"].copy()
    for col in ("Scarto % (exit fissa)", "Scarto % (completo)"):
        pw[col] = pw[col].map("{:+.1%}".format)
    _stampa("PWERM e modelli option-based (data di valutazione)", pw)

    art = t["Art60_ingresso"].copy()
    art = art[art.Configurazione.isin(["Exit fissa, senza leaver", "Exit incerta, con leaver"])]
    art["Valore / equity"] = art["Valore / equity"].map("{:.2%}".format)
    _stampa("Test art. 60 all'ingresso", art)

    _stampa("Convergenza del Monte Carlo (exit fissa, data di valutazione)", t["Convergenza_MC"])

    va = t["Varianti_MC_completo"].copy()
    va["Δ% dal caso base"] = va["Δ% dal caso base"].map("{:+.1%}".format)
    _stampa("Varianti con il Monte Carlo completo", va)
    lf = t["Leaver_fair_value"]
    lf = _percentuali(_percentuali(lf, ["Effetto dei leaver"], "{:+.1%}"), ["Valore / equity"], "{:.2%}")
    _stampa("Leaver: prezzo al valore intrinseco o al fair value (Monte Carlo, exit incerta)", lf)

    uscita = CARTELLA_RISULTATI / "04_montecarlo.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome, index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def _percentuali(df, colonne, formato="{:.1%}"):
    df = df.copy()
    for c in colonne:
        df[c] = df[c].map(lambda x: "" if pd.isna(x) else formato.format(x))
    return df


def passaggio_5(casi):
    print("\n" + "=" * 78 + "\nPASSAGGIO 5 – Modello strutturale\n" + "=" * 78)
    t = passaggio5.esegui(casi)

    cal = t["Calibrazione"]
    cal = cal[cal.Data == "valutazione"][["Caso", "σ attivo", "Volatilità dell'attivo", "V0 calibrato (€m)", "EV da business plan (€m)",
                                          "V0 / EV − 1", "Debito all'exit (€m)", "Volatilità equity implicita iniziale",
                                          "Volatilità equity del GBM", "Equity osservato (€m)", "Verifica MC: equity media scontata (€m)",
                                          "Errore standard verifica (€m)"]]
    _stampa("Calibrazione di V0 (data di valutazione)",
            _percentuali(cal, ["Volatilità dell'attivo", "V0 / EV − 1", "Volatilità equity implicita iniziale", "Volatilità equity del GBM"]))

    v = t["Valori"]
    for data in ("valutazione", "ingresso"):
        piv = v[v.Data == data].pivot_table(index=["Configurazione", "Modello"], columns="Caso",
                                            values="Valore del MIP (€m)", sort=False)
        _stampa(f"Valore del MIP (€m) alla data di {data}", piv)
    piv = v[v.Data == "valutazione"].pivot_table(index=["Configurazione", "Modello"], columns="Caso",
                                                 values="Δ% dal MC sull'equity", sort=False)
    _stampa("Differenza dal Monte Carlo sull'equity (data di valutazione)", piv.map("{:+.1%}".format))
    a = v[(v.Caso == "A") & (v.Data == "valutazione")].pivot_table(index="Configurazione", columns="Modello",
                                                                   values="Componente ratchet (€m)", sort=False)
    _stampa("Caso A, componente ratchet (€m, data di valutazione)", a)

    sc = t["Scomposizione"]
    _stampa("Scomposizione: strutturale (σ peer) − MC sull'equity (€m)",
            sc[sc.Data == "valutazione"].drop(columns="Data"))

    _stampa("Dispersione tra modelli con e senza il modello strutturale",
            _percentuali(t["Dispersione"], ["Dispersione senza strutturale", "Dispersione con strutturale (σ peer)"]))

    _stampa("Leva e volatilità implicita dell'equity (data di valutazione)",
            _percentuali(t["Dinamica_leva"], ["Leva attesa", "Volatilità equity implicita (strutturale)", "Volatilità equity (GBM)"]))

    art = _percentuali(t["Art60_ingresso"], ["Valore / equity"], "{:.2%}")
    art = art[art.Configurazione.isin(["Exit fissa, senza leaver", "Exit incerta, con leaver"])]
    _stampa("Test art. 60 all'ingresso", art)

    uscita = CARTELLA_RISULTATI / "05_strutturale.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome, index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def passaggio_6(casi):
    print("\n" + "=" * 78 + "\nPASSAGGIO 6 – Exit endogena\n" + "=" * 78)
    t = passaggio6.esegui(casi)
    v = t["Valori"]
    for data in ("valutazione", "ingresso"):
        x = v[v.Data == data].drop(columns=["Data"] + (["Fair value − prezzo (€m)"] if data == "valutazione" else []))
        x = _percentuali(_percentuali(x, ["Δ% da exit incerta"], "{:+.1%}"), ["Valore / equity"], "{:.2%}")
        _stampa(f"Valore del MIP con exit endogena alla data di {data} (€m)", x.astype(object).where(x.notna(), ""))
    _stampa("Distribuzione delle date di exit (data di valutazione)",
            _percentuali(t["Distribuzione_exit"], [c for c in t["Distribuzione_exit"].columns if c.isdigit()], "{:.0%}"))
    _stampa("Proventi del fondo: LSM ed exit fisse", t["Fondo_LSM_vs_fisse"])
    _stampa("Probabilità di exit vicino alle soglie (data di valutazione)",
            _percentuali(t["Conflitto_soglie"], ["P(exit | equity tra 90% e 100% della soglia)",
                                                 "P(exit | equity tra 100% e 110% della soglia)"]))
    uscita = CARTELLA_RISULTATI / "06_exit_endogena.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome, index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def passaggio_7(casi):
    print("\n" + "=" * 78 + "\nPASSAGGIO 7 – Confronto tra modelli\n" + "=" * 78)
    t = passaggio7.esegui(casi)

    v = t["Valori_per_modello"]
    piv = v.pivot_table(index="Modello", columns="Caso", values="Valore alla valutazione (€m)", sort=False)
    _stampa("Valore del MIP per modello (€m, data di valutazione)", piv)

    g = _percentuali(t["Dispersione_gruppi"], ["Dispersione"])
    _stampa("Dispersione per gruppi di modelli",
            g[["Caso", "Gruppo di modelli", "Ipotesi omogenee", "Modelli", "Minimo (€m)", "Massimo (€m)", "Dispersione",
               "Modello minimo", "Modello massimo"]])

    f = t["Fonti_di_divergenza"].copy()
    f["Effetto sul valore"] = [f"±{x:.1%}" if c == "intervallo" else f"{x:+.1%}" for x, c in zip(f["Effetto sul valore"], f["Confronto"])]
    _stampa("Fonti di divergenza (data di valutazione)", f)

    art = _percentuali(t["Art60_per_modello"], ["Fair value / equity"], "{:.2%}")
    _stampa("Art. 60: fair value / equity all'ingresso per modello", art)
    _stampa("Art. 60: sintesi per caso", _percentuali(t["Art60_sintesi"], ["Minimo", "Massimo"], "{:.2%}"))

    r = t["Ratchet_caso_A"].copy()
    r["Componente ratchet (€m)"] = [f"{x:.1%}" if m.startswith("Dispersione") else f"{x:.3f}"
                                    for x, m in zip(r["Componente ratchet (€m)"], r["Modello"])]
    _stampa("Caso A: componente ratchet per modello", r)

    uscita = CARTELLA_RISULTATI / "07_confronto_modelli.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome, index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def passaggio_8(casi):
    print("\n" + "=" * 78 + "\nPASSAGGIO 8 – Sensitivity\n" + "=" * 78)
    t = passaggio8.esegui(casi)

    def tornado(df, titolo, colonne):
        x = _percentuali(_percentuali(df, ["Effetto minimo", "Effetto massimo", "Effetto per unità"], "{:+.1%}"),
                         ["Ampiezza (max − min)"])
        _stampa(titolo, x[colonne].astype(object).where(x[colonne].notna(), ""))

    col = ["Caso", "Parametro", "Effetto minimo", "Livello minimo", "Effetto massimo", "Livello massimo", "Ampiezza (max − min)",
           "Effetto per unità", "Unità"]
    tp = t["Tornado_stima_plausibile"]
    for m in ("Monte Carlo completo", "OPM semplice", "PWERM da prassi"):
        tornado(tp[tp.Modello == m], f"Parametri di stima, intervallo plausibile – {m} (data di valutazione)", col)
    motivi = tp.drop_duplicates("Parametro")[["Parametro", "Intervallo"]]
    _stampa("Intervalli plausibili", motivi)

    tc = t["Tornado_contratto_struttura"]
    tornado(tc[tc.Modello == "Monte Carlo completo"],
            "Termini contrattuali e struttura del deal – Monte Carlo completo",
            ["Caso", "Parametro", "Tipo", "Effetto minimo", "Livello minimo", "Effetto massimo", "Livello massimo", "Ampiezza (max − min)"])

    _stampa("Parameter risk e model risk (data di valutazione)",
            _percentuali(t["Model_vs_parameter_risk"], ["Model risk: OPM semplice vs modelli completi (gruppo 2)",
                                                        "Model risk: + strutturale (gruppo 3)", "Sua ampiezza (intervallo plausibile)"]))

    cl = t["Classifica_modello_parametri"]
    cl = _percentuali(cl[cl.Rango <= 10], ["Scostamento dal caso base"])
    _stampa("Classifica: scelte di modello e parametri di stima (prime 10 per caso)", cl)

    art = _percentuali(t["Art60_sensitivity"], ["Caso base", "Minimo", "Denominatore alternativo (senza shareholder loan), caso base"], "{:.2%}")
    _stampa("Art. 60 all'ingresso: sensitivity", art)

    cv = t["Convenzione_volatilita"]
    cv = _percentuali(cv, ["Volatilità relevered – valutazione", "Volatilità dei peer senza relevering – valutazione"])
    cv = _percentuali(cv, ["Δ% dal caso base"], "{:+.1%}")
    cv = _percentuali(cv, ["Fair value / equity all'ingresso – relevered", "Fair value / equity all'ingresso – senza relevering"], "{:.2%}")
    _stampa("Convenzione di volatilità: peer senza relevering invece di relevered", cv)

    _stampa("Impostazioni numeriche (data di valutazione)",
            _percentuali(t["Impostazioni_numeriche"], ["Ampiezza / riferimento"]))

    va = t["Varianti_V1_V8"]
    _stampa("Varianti V1-V8 (data di valutazione)",
            _percentuali(va, [c for c in va.columns if c.startswith("Δ%")], "{:+.1%}"))

    uscita = CARTELLA_RISULTATI / "08_sensitivity.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome[:31], index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def passaggio_9(casi):
    print("\n" + "=" * 78 + "\nPASSAGGIO 9 – Tasso di sconto del PWERM e mappa condizionata\n" + "=" * 78)
    t = passaggio9.esegui(casi)

    ts = t["Tasso_sconto_PWERM"]
    tassi = ["Ke (equity)", "Risk-free annuo", "Rendimento richiesto dal MIP (CAPM con Ω)",
             "Tasso implicito (distribuzione del modello)", "Tasso implicito (scenari di prassi)"]
    _stampa("Elasticità del MIP e tasso di sconto",
            _percentuali(ts[["Data", "Caso", "Valore OPM (€m)", "Delta (Δ)", "Elasticità Ω = Δ·E/V", *tassi,
                             "PWERM da prassi al Ke (€m)"]], tassi))
    _stampa("PWERM al Ke e al rendimento richiesto dal MIP (€m)",
            ts[["Data", "Caso", "Valore OPM (€m)", "PWERM da quantili (limite) al Ke (€m)",
                "PWERM da quantili (limite) al rendimento del MIP (€m)", "PWERM da prassi al Ke (€m)",
                "PWERM da prassi al rendimento del MIP (€m)"]])

    _stampa("Mappa condizionata: sintesi",
            _percentuali(t["Sintesi_mappa"], [c for c in t["Sintesi_mappa"].columns if c not in ("Caso", "Dispersione")]))
    for nome, titolo in (("Griglia_vol_soglie_g3", "Gruppo 3 (OPM, MC completo, strutturale): volatilità × soglie"),
                         ("Griglia_vol_soglie_g2", "Gruppo 2 (OPM vs MC completo): volatilità × soglie"),
                         ("Griglia_vol_exit_g2", "Gruppo 2 (OPM vs MC completo): volatilità × data di exit")):
        g = t[nome]
        _stampa(titolo, _percentuali(g, [c for c in g.columns if c not in ("Caso", "Δ volatilità")]))

    uscita = CARTELLA_RISULTATI / "09_approfondimenti.xlsx"
    with pd.ExcelWriter(uscita) as xw:
        foglio_info().to_excel(xw, sheet_name="Info", index=False)
        for nome, df in t.items():
            df.to_excel(xw, sheet_name=nome[:31], index=False)
    print(f"\nRisultati salvati in: {uscita.relative_to(CARTELLA_RISULTATI.parent)}")


def passaggio_10(casi):
    print("\n" + "=" * 78 + "\nPASSAGGIO 10 – Grafici e tabelle\n" + "=" * 78)
    t = passaggio10.esegui(casi)
    _stampa("Grafici prodotti", t["Index"][["Figure", "Short title"]])
    d = t["T5_Payoff_distribution"]
    _stampa("Distribuzione del payoff (multipli del prezzo pagato)",
            _percentuali(d, [c for c in d.columns if "(%)" in c]))
    t6 = t["T6_Discount_rate"]
    _stampa("PWERM al Ke e al rendimento richiesto dal MIP (€m, data di valutazione)",
            t6[t6.Date == "Valuation date"][["Case", "OPM value (€m)", "PWERM full distribution at Ke (€m)",
                                             "PWERM full distribution at MIP return (€m)", "PWERM practice scenarios at Ke (€m)",
                                             "PWERM practice scenarios at MIP return (€m)"]])
    passaggio10.salva_excel(t)
    print(f"\nGrafici salvati in: {passaggio10.CARTELLA_GRAFICI.relative_to(CARTELLA_RISULTATI.parent)}")
    print(f"Tabelle salvate in: {passaggio10.FILE_TABELLE.relative_to(CARTELLA_RISULTATI.parent)}")


def main():
    CARTELLA_RISULTATI.mkdir(exist_ok=True)
    scelta = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    intestazione()
    casi = carica_casi()
    if "1" in scelta:
        passaggio_1(casi)
    if "2" in scelta:
        passaggio_2(casi)
    if "3" in scelta:
        passaggio_3(casi)
    if "4" in scelta:
        passaggio_4(casi)
    if "5" in scelta:
        passaggio_5(casi)
    if "6" in scelta:
        passaggio_6(casi)
    if "7" in scelta:
        passaggio_7(casi)
    if "8" in scelta:
        passaggio_8(casi)
    if "9" in scelta:
        passaggio_9(casi)
    if "10" in scelta:
        passaggio_10(casi)


if __name__ == "__main__":
    main()
