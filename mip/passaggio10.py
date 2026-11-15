# Passaggio 10: grafici e tabelle.
from __future__ import annotations

import datetime as dt
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .config import CARTELLA_RISULTATI, FILE_INPUT, FONT_GRAFICI
from .input import Caso, hash_file
from .passaggio9 import tabella_tasso_sconto
from .waterfall import payoff, soglie

CARTELLA_GRAFICI = CARTELLA_RISULTATI / "grafici"
FILE_TABELLE = CARTELLA_RISULTATI / "10_thesis_tables.xlsx"

COLORE = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TESTO, TESTO2, GRIGLIA, NEUTRO = "#0b0b0b", "#52514e", "#ebeae6", "#a9a8a3"
NOME = {"A": "Case A – cliff with ratchet", "B": "Case B – hurdle with catch-up", "C": "Case C – pay-to-play"}

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_SOSTITUTI = {"Times New Roman": "Liberation Serif", "Arial": "Liberation Sans"}

plt.rcParams.update({
    "font.family": [FONT_GRAFICI, _SOSTITUTI.get(FONT_GRAFICI, "DejaVu Sans"), "DejaVu Sans"], "font.size": 8.8, "axes.titlesize": 9.4, "axes.titleweight": "semibold",
    "axes.titlepad": 6, "axes.titlelocation": "left", "axes.labelsize": 8.8, "axes.labelpad": 5,
    "axes.edgecolor": NEUTRO, "axes.linewidth": 0.8, "axes.labelcolor": TESTO2, "axes.titlecolor": TESTO,
    "xtick.color": TESTO2, "ytick.color": TESTO2, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "axes.grid": False, "grid.color": GRIGLIA, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.2, "legend.handlelength": 1.6, "legend.columnspacing": 1.6,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "lines.linewidth": 1.8, "hatch.linewidth": 0.7,
    "svg.fonttype": "path", "pdf.fonttype": 42,
})
LARGHEZZA = 6.3

MODELLI = {
    "OPM semplice (exit fissa, senza leaver)": "Simple OPM (fixed exit, no leavers)",
    "OPM semplice": "Simple OPM",
    "Albero binomiale completo": "Binomial tree (full)",
    "Albero trinomiale completo": "Trinomial tree (full)",
    "Monte Carlo completo": "Monte Carlo (full)",
    "Strutturale completo (σ attivo dei peer)": "Structural (peer asset σ)",
    "Strutturale completo (σ attivo coerente)": "Structural (consistent asset σ)",
    "PWERM da prassi": "PWERM (practice scenarios)",
    "PWERM da quantili (diagnostico)": "PWERM (quantiles, diagnostic)",
    "Exit endogena – regola obiettivi": "Endogenous exit – target rule",
    "Exit endogena – LSM (ottimo del fondo)": "Endogenous exit – LSM (fund-optimal)",
}
BREVE = {
    "Simple OPM (fixed exit, no leavers)": "Simple OPM", "Binomial tree (full)": "Binomial tree",
    "Trinomial tree (full)": "Trinomial tree", "Monte Carlo (full)": "Monte Carlo",
    "Structural (consistent asset σ)": "Structural (consistent σ)", "Structural (peer asset σ)": "Structural (peer σ)",
    "PWERM (practice scenarios)": "PWERM (practice)", "Endogenous exit – target rule": "Endog. exit – target rule",
    "Endogenous exit – LSM (fund-optimal)": "Endog. exit – LSM", "PWERM (quantiles, diagnostic)": "PWERM (quantiles, diagn.)",
}
ORDINE = list(BREVE.values())
FAMIGLIE = {"option-based semplice": "simple option-based", "option-based completo": "full option-based",
            "strutturale": "structural", "PWERM": "PWERM", "diagnostico": "diagnostic", "exit endogena": "endogenous exit"}
GRUPPI = {"1": "1. Full option-based (trees, Monte Carlo): same assumptions, different engines",
          "2": "2. + simple OPM (practice: fixed exit, no leavers)", "3": "3. + structural model",
          "4": "4. + practice PWERM", "5": "5. + endogenous exit", "6": "6. + quantile PWERM (diagnostic)"}
PARAMETRI = {"Volatilità equity": "Equity volatility", "DLOM": "DLOM",
             "Finestra di stima della volatilità": "Volatility estimation window",
             "Probabilità annua di uscita (leaver)": "Annual leaver probability", "Data di exit": "Exit date",
             "Risk-free": "Risk-free rate", "Definizione di leva": "Leverage definition",
             "Convenzione di volatilità": "Peer volatility, not relevered"}
SI_NO = {"Sì": "Yes", "No": "No"}
DATE = {"valutazione": "Valuation date", "ingresso": "Entry date"}


def _modello(m):
    return MODELLI.get(m, m)


def _livello(x):
    s = str(x)
    for a, b in (("Solo debito netto (base)", "Net debt only (base)"), ("Debito netto + debt-like", "Net debt + debt-like"),
                 (" anni", " years"), (" anno", " year"), ("anni", "years")):
        s = s.replace(a, b)
    return s


def _pct(x, d=0):
    return f"{x:.{d}%}"


def _asse_pct(ax, asse="y", d=0):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: _pct(v, d))
    (ax.yaxis if asse == "y" else ax.xaxis).set_major_formatter(f)


def _salva(fig, nome):
    legende = getattr(fig, "_legende", [])
    if legende:
        H = fig.get_figheight()
        righe = [math.ceil(len(l) / n) for _, l, n in legende]
        riga, gap = 0.18, 0.08
        spazio = (gap + riga * sum(righe) + 0.06 * (len(legende) - 1)) / H
        fig.tight_layout(rect=(0, spazio, 1, 1))
        y = spazio - gap / H * 0.6
        for (h, l, n), r in zip(legende, righe):
            fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncols=n, frameon=False)
            y -= (riga * r + 0.06) / H
    CARTELLA_GRAFICI.mkdir(parents=True, exist_ok=True)
    fig.savefig(CARTELLA_GRAFICI / f"{nome}.png", facecolor="white")
    for fmt in ("svg", "pdf"):
        (CARTELLA_GRAFICI / fmt).mkdir(exist_ok=True)
        fig.savefig(CARTELLA_GRAFICI / fmt / f"{nome}.{fmt}", facecolor="white")
    plt.close(fig)


def _senza_asse_y(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)


def _etichetta(ax, x, y, testo, dx=5, ha="left", **kw):
    ax.annotate(testo, (x, y), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                fontsize=kw.pop("fontsize", 6.8), color=kw.pop("color", TESTO2), zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5), **kw)


def _leggi(file, foglio):
    p = CARTELLA_RISULTATI / file
    if not p.exists():
        raise FileNotFoundError(f"Manca {p.name}: esegui prima il passaggio {file[:2].lstrip('0')}.")
    return pd.read_excel(p, sheet_name=foglio)


def _legenda_sotto(fig, handles, labels, ncols, y=None):
    fig._legende = getattr(fig, "_legende", []) + [(handles, labels, ncols)]


def fig01_profilo(casi):
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 2.4), sharey=True)
    dati = []
    for ax, (k, c) in zip(assi, casi.items()):
        t = c.t_exit
        sg = soglie(c, t)
        H = float(sg["hurdle"])
        E = np.linspace(0, 2.5 * H, 600)
        v = np.asarray(payoff(c, E, t)["mip"], dtype=float)
        ax.plot(E, v, color=COLORE[k])
        ax.axvline(H, color=TESTO2, lw=1, ls="--")
        ax.text(H * 0.97, 0.97, "hurdle", color=TESTO2, fontsize=8.2, va="top", ha="right",
                transform=ax.get_xaxis_transform())
        altra = {"A": ("ratchet", "ratchet"), "B": ("fine_catch_up", "end of\ncatch-up")}.get(k)
        if altra:
            R_ = float(sg[altra[0]])
            ax.axvline(R_, color=TESTO2, lw=1, ls=":")
            ax.text(R_ + 0.03 * E.max(), 0.97, altra[1], color=TESTO2, fontsize=8.2, va="top",
                    transform=ax.get_xaxis_transform())
        ax.set_title(NOME[k].replace(" – ", "\n"))
        ax.set_xlabel("Equity value at exit (€m)")
        dati += [{"Case": k, "Equity value at exit (€m)": e, "MIP pool payoff (€m)": x} for e, x in zip(E[::20], v[::20])]
    assi[0].set_ylabel("MIP pool payoff (€m)")
    assi[0].set_ylim(0, max(d_["MIP pool payoff (€m)"] for d_ in dati) * 1.2)
    fig.tight_layout()
    _salva(fig, "Fig01_payoff_profile")
    return pd.DataFrame(dati)


MISURE = {"real": "Real-world (drift at Ke)", "rn": "Risk-neutral (drift at risk-free)"}
FASCE = [("Zero", None, 0.0), ("0–1x\n(loss)", 0.0, 1.0), ("1–3x", 1.0, 3.0), ("3–10x", 3.0, 10.0), ("> 10x", 10.0, np.inf)]


def tabella_distribuzione(casi, n=200_000, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    righe, fasce = [], []
    for k, c in casi.items():
        T, s, E0 = c.T, c.vol_valutazione, c.equity_valutazione
        for chiave, mu in (("real", math.log(1 + c.ke_valutazione)), ("rn", c.rf_cont_valutazione)):
            ET = E0 * np.exp((mu - s * s / 2) * T + s * math.sqrt(T) * z)
            x = np.asarray(payoff(c, ET, c.t_exit)["mip"], dtype=float) / c.importo_mip
            zero = x <= 1e-12
            righe.append({"Case": k, "Measure": MISURE[chiave], "P(MIP = 0) (%)": float(zero.mean()),
                          "P(payoff < price paid) (%)": float(np.mean(x < 1)), "Median (x price)": float(np.median(x)),
                          "75th percentile (x price)": float(np.quantile(x, 0.75)),
                          "90th percentile (x price)": float(np.quantile(x, 0.90)),
                          "Mean (x price)": float(x.mean()),
                          "Share of expected payoff from best 10% (%)": float(np.sort(x)[-n // 10:].sum() / x.sum())})
            if chiave == "real":
                for nome, lo, hi in FASCE:
                    m = zero if lo is None else (~zero) & (x > lo) & (x <= hi)
                    fasce.append({"Case": k, "Payoff bucket": nome.replace("\n", " "), "Probability (%)": float(m.mean()),
                                  "Share of expected payoff (%)": float(x[m].sum() / x.sum())})
    return pd.DataFrame(righe), pd.DataFrame(fasce)


def fig02_distribuzione(fasce):
    x = np.arange(len(FASCE)) * 1.25
    w = 0.32
    for col, nome, etichetta in (("Probability (%)", "Fig02a_payoff_probability", "Probability"),
                                 ("Share of expected payoff (%)", "Fig02b_payoff_value_share", "Share of the expected payoff")):
        fig, ax = plt.subplots(figsize=(LARGHEZZA, 2.6))
        for i, k in enumerate("ABC"):
            v = fasce[fasce.Case == k][col].values
            ax.bar(x + (i - 1) * w, v, w * 0.9, color=COLORE[k], label=f"Case {k}")
            for xi, vi in zip(x, v):
                if vi >= 0.005:
                    ax.annotate(f"{vi * 100:.0f}%", (xi + (i - 1) * w, vi), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.3, color=TESTO2)
        ax.set_xticks(x, [f[0].replace("\n", " ") for f in FASCE])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 0.75)
        _asse_pct(ax)
        ax.set_ylabel(etichetta)
        ax.set_xlabel("Payoff at exit as a multiple of the price paid by management")
        h, l = ax.get_legend_handles_labels()
        _legenda_sotto(fig, h, l, 3)
        _salva(fig, nome)


def fig03_valori(v):
    v = v.assign(M=v.Modello.map(_modello).map(BREVE))
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 3.4), sharey=True)
    y = np.arange(len(ORDINE))[::-1]
    for ax, k in zip(assi, "ABC"):
        d = v[v.Caso == k].set_index("M").reindex(ORDINE)
        val = d["Valore alla valutazione (€m)"].values
        diagn = np.array([m.endswith("diagn.)") for m in ORDINE])
        rif = d.loc["Monte Carlo", "Valore alla valutazione (€m)"]
        ax.axvline(rif, color=TESTO, lw=0.9, ls=":", zorder=1)
        ax.barh(y[~diagn], val[~diagn], height=0.62, color=COLORE[k], zorder=2)
        ax.barh(y[diagn], val[diagn], height=0.62, color="white", edgecolor=COLORE[k], hatch="////", linewidth=0.9, zorder=2)
        for yi, x in zip(y, val):
            if x == x:
                _etichetta(ax, x, yi, f"{x:.1f}", color=TESTO)
            else:
                _etichetta(ax, 0, yi, "not applied", dx=3)
        ax.set_title(f"Case {k}")
        ax.set_xlim(0, v["Valore alla valutazione (€m)"].max() * 1.25)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("€m")
    assi[0].set_yticks(y, ORDINE)
    h = [Line2D([], [], color=TESTO, lw=0.9, ls=":"), Patch(facecolor="white", edgecolor=TESTO2, hatch="////")]
    _legenda_sotto(fig, h, ["Monte Carlo (full) value", "diagnostic only"], 2)
    _salva(fig, "Fig03_values_by_model")


def fig04_gruppi(g):
    fig, ax = plt.subplots(figsize=(LARGHEZZA, 2.9))
    gruppi = g["Gruppo di modelli"].unique()
    x = np.arange(len(gruppi))
    w = 0.26
    for i, k in enumerate("ABC"):
        d = g[g.Caso == k].set_index("Gruppo di modelli").reindex(gruppi)
        om = (d["Ipotesi omogenee"] == "Sì").values
        disp = d["Dispersione"].values
        ax.bar(x[om] + (i - 1) * w, disp[om], w * 0.92, color=COLORE[k], label=f"Case {k}")
        ax.bar(x[~om] + (i - 1) * w, disp[~om], w * 0.92, color="white", edgecolor=COLORE[k], hatch="////", linewidth=1)
    ax.set_xticks(x, ["1. Full models\n(trees, MC)", "2. + Simple\nOPM", "3. + Struct.\nmodel",
                      "4. + Practice\nPWERM", "5. + Endog.\nexit", "6. + Quantile\nPWERM"][:len(gruppi)], fontsize=8)
    _asse_pct(ax)
    ax.set_ylabel("Dispersion = (max − min) / median")
    ax.grid(axis="y")
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="white", edgecolor=TESTO2, hatch="////")]
    l += ["indicative – different assumptions"]
    _legenda_sotto(fig, h, l, 4)
    _salva(fig, "Fig04_dispersion_by_group")


def fig05_convergenza(alberi, mc):
    fig, assi = plt.subplots(2, 2, figsize=(LARGHEZZA, 4.6), sharey=True)
    cols = [c for c in alberi.columns if c.startswith("N=")]
    N = [int(c[2:]) for c in cols]
    dati = []
    for ax, (tipo, titolo) in zip(assi[0], (("binomiale", "Binomial tree"), ("trinomiale", "Trinomial tree"))):
        for k in "ABC":
            r = alberi[(alberi.Caso == k) & (alberi.Albero == tipo)].iloc[0]
            err = [abs(r[c] / r["Black-Scholes"] - 1) for c in cols]
            ax.plot(N, err, color=COLORE[k], marker="o", ms=3.5, lw=1.5)
            dati += [{"Panel": titolo, "Case": k, "Steps or paths": n, "Relative error (%)": e} for n, e in zip(N, err)]
        ax.set_title(titolo, loc="left")
        ax.set_xlabel("Tree steps (N)")
    assi[0][0].set_ylabel("|value / Black-Scholes − 1|")
    for ax, (col, titolo) in zip(assi[1], (("Errore standard semplice", "Monte Carlo, plain"),
                                           ("Errore standard con controllo", "Monte Carlo with control variate"))):
        for k in "ABC":
            d = mc[mc.Caso == k]
            e = (d[col] / d["Black-Scholes"]).values
            dati += [{"Panel": titolo, "Case": k, "Steps or paths": n, "Relative error (%)": v} for n, v in zip(d.Simulazioni, e)]
            if (e > 1e-8).all():
                ax.plot(d.Simulazioni, e, color=COLORE[k], marker="o", ms=3.5, lw=1.5)
        ax.set_title(titolo, loc="left")
        ax.set_xlabel("Simulated paths")
    assi[1][0].set_ylabel("standard error / value")
    for ax in assi.flat:
        ax.grid(True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(5e-6, 8e-2)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.tight_layout()
    _legenda_sotto(fig, [Line2D([], [], color=COLORE[k], lw=2) for k in "ABC"], ["Case A", "Case B", "Case C"], 3)
    _salva(fig, "Fig05_numerical_convergence")
    return pd.DataFrame(dati)


def fig06_art60(a):
    a = a.assign(M=a.Modello.map(_modello).map(BREVE))
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 3.4), sharey=True)
    y = np.arange(len(ORDINE))[::-1]
    for ax, k in zip(assi, "ABC"):
        d = a[a.Caso == k].set_index("M").reindex(ORDINE)
        x = d["Fair value / equity"].values
        sotto = x < 0.01
        ax.axvline(0.01, color=TESTO, lw=1.1, zorder=2)
        ax.scatter(x[~sotto], y[~sotto], s=26, color=COLORE[k], zorder=3)
        ax.scatter(x[sotto], y[sotto], s=26, facecolor="white", edgecolor=COLORE[k], linewidth=1.4, zorder=3)
        xmax = a["Fair value / equity"].max() * 1.35
        for xi, yi in zip(x, y):
            if xi == xi:
                _etichetta(ax, xi, yi, _pct(xi, 2), dx=7)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02 if xmax > 0.03 else 0.01))
        _asse_pct(ax, "x")
        ax.set_title(f"Case {k}")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    assi[0].set_yticks(y, ORDINE)
    assi[1].set_xlabel("MIP fair value / total investment at entry")
    h = [Line2D([], [], color=TESTO, lw=1.1), Line2D([], [], ls="", marker="o", color=TESTO2),
         Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=TESTO2, markeredgewidth=1.4)]
    _legenda_sotto(fig, h, ["1% threshold", "test passed", "test failed"], 3)
    _salva(fig, "Fig06_art60_by_model")


def tabella_margine(a):
    a = a.assign(Model=a.Modello.map(_modello))
    a = a.assign(**{"Headroom (%)": 1 - 0.01 / a["Fair value / equity"]})
    t = a.pivot_table(index="Model", columns="Caso", values="Headroom (%)", sort=False)
    t.columns = [f"Case {c} – headroom (%)" for c in t.columns]
    return t.reset_index()


def tabella_ampiezza_margine(a):
    a = a[~a.Modello.str.contains("diagnostico")].assign(H=lambda d: 1 - 0.01 / d["Fair value / equity"])
    righe = []
    for k, d in a.groupby("Caso"):
        for nome, sel in (("all valuation candidates", d), ("excluding endogenous exit",
                                                            d[~d.Modello.str.startswith("Exit endogena")])):
            lo, hi = sel.loc[sel.H.idxmin()], sel.loc[sel.H.idxmax()]
            righe.append({"Case": k, "Model set": nome, "Lowest headroom (%)": lo.H, "Model (lowest)": _modello(lo.Modello),
                          "Highest headroom (%)": hi.H, "Model (highest)": _modello(hi.Modello),
                          "Spread (percentage points)": (hi.H - lo.H) * 100})
    return pd.DataFrame(righe)


def fig06b_margine(a):
    a = a[~a.Modello.str.contains("diagnostico")].assign(H=lambda d: 1 - 0.01 / d["Fair value / equity"])
    fig, ax = plt.subplots(figsize=(LARGHEZZA, 2.5))
    for i, k in enumerate("ABC"):
        d = a[a.Caso == k]
        y = 2 - i
        lo, hi = d.loc[d.H.idxmin()], d.loc[d.H.idxmax()]
        ax.plot([lo.H, hi.H], [y, y], color=COLORE[k], lw=10, alpha=0.22, solid_capstyle="round", zorder=1)
        ax.scatter(d.H, [y] * len(d), s=20, color=COLORE[k], zorder=3)
        mc = d[d.Modello == "Monte Carlo completo"].H.iloc[0]
        ax.scatter([mc], [y], s=56, facecolor="white", edgecolor=TESTO, linewidth=1.2, zorder=4)
        ax.annotate(BREVE[_modello(lo.Modello)], (lo.H, y), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.3, color=TESTO2)
        ax.annotate(BREVE[_modello(hi.Modello)], (hi.H, y), xytext=(9, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.3, color=TESTO2)
        ax.text(1.02, y, f"{(hi.H - lo.H) * 100:.0f} pts", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.8, fontweight="semibold", color=TESTO)
    ax.axvline(0, color=TESTO, lw=1.1)
    ax.annotate("← test failed", (0, 2.62), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                fontsize=7.5, color=TESTO2)
    ax.annotate("test passed →", (0, 2.62), xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                fontsize=7.5, color=TESTO2)
    ax.text(1.02, 2.62, "spread", transform=ax.get_yaxis_transform(), fontsize=7.7, color=TESTO2, va="center")
    ax.set_yticks([2, 1, 0], ["Case A", "Case B", "Case C"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 2.85)
    ax.set_xlim(-0.75, 1.45)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    _asse_pct(ax, "x")
    ax.set_xlabel("Headroom to the 1% threshold: fall in the MIP fair value that would bring it to 1% of the total investment")
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], ls="", marker="o", color=TESTO2, ms=5),
         Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=TESTO, ms=7)]
    _legenda_sotto(fig, h, ["one dot per valuation model", "full Monte Carlo"], 2)
    _salva(fig, "Fig06b_art60_headroom")


def _etichetta_parametro(p):
    return {"Annual leaver probability": "Leaver probability", "Volatility estimation window": "Volatility window"}.get(p, p)


def _chiaro(colore, f=0.55):
    r, g, b = matplotlib.colors.to_rgb(colore)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def fig07_parametri_vs_modello(t, mvp, valori, lfv):
    t = t[t.Modello == "Monte Carlo completo"].assign(P=lambda d: d.Parametro.map(PARAMETRI))
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 3.6))
    for ax, k in zip(assi, "ABC"):
        r = mvp[mvp.Caso == k].iloc[0]
        opm = valori[(valori.Caso == k) & valori.Modello.str.startswith("OPM semplice")]["Valore alla valutazione (€m)"].iloc[0]
        fv = lfv[(lfv.Data == "valutazione") & (lfv.Caso == k) & lfv["Prezzo pagato al leaver"].str.startswith("Fair value")]["Valore del MIP (€m)"].iloc[0]
        voci = [("Model: + structural", r["Model risk: + strutturale (gruppo 3)"], True),
                ("Model: simple vs full", r["Model risk: OPM semplice vs modelli completi (gruppo 2)"], True),
                ("Model: simple vs full,\nleavers at option value", opm / fv - 1, True)]
        voci += [(_etichetta_parametro(p), v, False)
                 for p, v in zip(t[t.Caso == k].P, t[t.Caso == k]["Ampiezza (max − min)"])]
        voci.sort(key=lambda x: x[1])
        y = np.arange(len(voci))
        colori = [COLORE[k] if m else _chiaro(COLORE[k]) for _, _, m in voci]
        ax.barh(y, [v for _, v, _ in voci], height=0.66, color=colori)
        for yi, (_, v, m) in zip(y, voci):
            _etichetta(ax, v, yi, _pct(v), dx=3, color=TESTO if m else TESTO2,
                       fontweight="semibold" if m else "normal")
        ax.set_yticks(y, [n for n, _, _ in voci], fontsize=7.2)
        for lab, (_, _, m) in zip(ax.get_yticklabels(), voci):
            lab.set_fontweight("semibold" if m else "normal")
            lab.set_color(TESTO if m else TESTO2)
        ax.set_xlim(0, 0.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
        _asse_pct(ax, "x")
        ax.set_title(f"Case {k}")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    assi[1].set_xlabel("Effect on the MIP value (range across models or across the plausible parameter interval)")
    h = [Patch(color=TESTO2), Patch(color=_chiaro(TESTO2))]
    _legenda_sotto(fig, h, ["model choice (dispersion between models)",
                            "estimation parameter (full Monte Carlo)"], 2)
    _salva(fig, "Fig07_parameter_vs_model_risk")


def fig08_tornado(t):
    t = t[t.Modello == "Monte Carlo completo"].assign(P=lambda d: d.Parametro.map(PARAMETRI))
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 2.8), sharey=True)
    parametri = t[t.Caso == "A"].sort_values("Ampiezza (max − min)").P.tolist()
    y = np.arange(len(parametri))
    for ax, k in zip(assi, "ABC"):
        d = t[t.Caso == k].set_index("P").reindex(parametri)
        lo, hi = d["Effetto minimo"].values, d["Effetto massimo"].values
        ax.barh(y, np.minimum(lo, 0), height=0.62, color=COLORE[k], alpha=0.5)
        ax.barh(y, np.maximum(hi, 0), height=0.62, color=COLORE[k])
        ax.axvline(0, color=TESTO, lw=0.8)
        ax.set_xlim(-0.3, 0.3)
        _asse_pct(ax, "x")
        ax.set_title(f"Case {k}", loc="left")
        ax.grid(axis="x")
    assi[0].set_yticks(y, [_etichetta_parametro(p) for p in parametri])
    assi[1].set_xlabel("Change in value from the base case (plausible interval, full Monte Carlo)")
    h = [Patch(color=TESTO2, alpha=0.5), Patch(color=TESTO2)]
    _legenda_sotto(fig, h, ["value decreases", "value increases"], 2)
    fig.tight_layout()
    _salva(fig, "Fig08_tornado")


def fig09_art60_sensitivity(s):
    col = "Fair value / equity all'ingresso"
    d = s[(s.Caso == "B") & (s.Tipo == "stima") & (s["Nell'intervallo plausibile"] == "Sì") & s[col].notna()
          & (s.Parametro != "Definizione di leva")]
    cv = s[(s.Caso == "B") & (s.Parametro == "Convenzione di volatilità") & (s["Caso base"] != "Sì")]
    d = pd.concat([d, cv])
    d = d[d.Modello.isin(["OPM semplice", "Monte Carlo completo"])].assign(P=lambda x: x.Parametro.map(PARAMETRI))
    parametri = list(dict.fromkeys(d.P))
    fig, ax = plt.subplots(figsize=(LARGHEZZA, 3.0))
    for j, (m, colore, et) in enumerate((("OPM semplice", SERIE[3], "Simple OPM"),
                                         ("Monte Carlo completo", SERIE[1], "Monte Carlo (full)"))):
        dm = d[d.Modello == m]
        y = np.array([parametri.index(p) for p in dm.P]) + (0.16 if j == 0 else -0.16)
        x = dm[col].values
        sotto = x < 0.01
        ax.scatter(x[~sotto], y[~sotto], s=24, color=colore, zorder=3, label=et)
        ax.scatter(x[sotto], y[sotto], s=24, facecolor="white", edgecolor=colore, linewidth=1.5, zorder=3)
    ax.axvline(0.01, color=TESTO, lw=1.2)
    ax.axhline(len(parametri) - 1.5, color=NEUTRO, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(range(len(parametri)), [_etichetta_parametro(p) for p in parametri])
    ax.get_yticklabels()[-1].set_fontstyle("italic")
    ax.invert_yaxis()
    _asse_pct(ax, "x", 2)
    ax.set_xlabel("MIP fair value / total investment at entry")
    ax.grid(axis="x")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=TESTO, lw=1.2),
          Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=TESTO2, markeredgewidth=1.5)]
    l += ["1% threshold", "test failed"]
    _legenda_sotto(fig, h, l, 4)
    _salva(fig, "Fig09_art60_sensitivity_caseB")
    out = d[["P", "Livello", "Modello", col]].copy()
    out.columns = ["Parameter", "Level", "Model", "MIP fair value / equity at entry (%)"]
    out["Level"] = out["Level"].map(_livello)
    out["Model"] = out["Model"].map(_modello)
    return out


def tabella_convenzione(cv):
    t = cv.assign(Model=cv.Modello.map(_modello))
    return pd.DataFrame({"Case": t.Caso, "Model": t.Model,
                         "Equity volatility, relevered (%)": t["Volatilità relevered – valutazione"],
                         "Equity volatility, peer median not relevered (%)": t["Volatilità dei peer senza relevering – valutazione"],
                         "Value, relevered (€m)": t["Valore relevered (€m)"],
                         "Value, not relevered (€m)": t["Valore senza relevering (€m)"],
                         "Change in value (%)": t["Δ% dal caso base"],
                         "FV / total investment at entry, relevered (%)": t["Fair value / equity all'ingresso – relevered"],
                         "FV / total investment at entry, not relevered (%)": t["Fair value / equity all'ingresso – senza relevering"],
                         "1% test passed, not relevered": t["Soglia 1% rispettata – senza relevering"].map(SI_NO)})


def _mappa(griglia, colonne, etichette_x, nome, asse_x, colonna_base):
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 2.6), sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blu", ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"])
    valori = griglia[colonne].values
    lo, hi = float(np.nanmin(valori)), float(np.nanmax(valori))
    for ax, k in zip(assi, "ABC"):
        d = griglia[griglia.Caso == k].sort_values("Δ volatilità", ascending=False)
        z = d[colonne].values
        im = ax.pcolormesh(z, cmap=cmap, vmin=lo, vmax=hi, edgecolors="white", linewidth=1.5)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                ax.text(j + 0.5, i + 0.5, _pct(z[i, j], 1), ha="center", va="center", fontsize=7.5, color=TESTO)
        i0 = list(d["Δ volatilità"]).index(0.0)
        ax.add_patch(matplotlib.patches.Rectangle((colonna_base, i0), 1, 1, fill=False, edgecolor=TESTO, lw=1.4))
        ax.set_xticks(np.arange(len(colonne)) + 0.5, etichette_x)
        ax.set_yticks(np.arange(len(d)) + 0.5, [f"{v * 100:+.0f} pts" if v else "base" for v in d["Δ volatilità"]])
        ax.invert_yaxis()
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"Case {k}", loc="left")
    assi[0].set_ylabel("Change in equity volatility")
    assi[1].set_xlabel(asse_x)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=list(assi), orientation="horizontal", fraction=0.05, pad=0.32, aspect=45)
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _pct(v)))
    cb.set_label("Dispersion between models = (max − min) / median   ·   framed cell: base case", fontsize=8, color=TESTO2)
    cb.outline.set_visible(False)
    _salva(fig, nome)


def fig10_mappa_g3(g3):
    _mappa(g3, [c for c in g3.columns if c.startswith("Fattore soglie")], ["× 0.8", "× 1.0", "× 1.2"],
           "Fig10_conditional_map_group3", "Hurdle level (multiple of the base-case level)", 1)


def fig11_mappa_exit(g2):
    _mappa(g2, [c for c in g2.columns if c.startswith("Δ exit")], ["−1", "−0.5", "0", "+0.5", "+1"],
           "Fig11_conditional_map_exit", "Shift of the exit date (years)", 2)


def fig12_tassi(ts):
    d = ts[ts.Data == "valutazione"].set_index("Caso")
    serie = [("Ke (equity)", "Equity cost of capital (Ke)"),
             ("Rendimento richiesto dal MIP (CAPM con Ω)", "MIP required return: rf + Ω(Ke − rf)"),
             ("Tasso implicito (distribuzione del modello)", "Rate implied by the option model")]
    fig, ax = plt.subplots(figsize=(LARGHEZZA, 2.6))
    x = np.arange(3)
    w = 0.24
    for i, (c, et) in enumerate(serie):
        v = d.loc[list("ABC"), c].values
        ax.bar(x + (i - 1) * w, v, w * 0.92, color=SERIE[i], label=et)
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 1) * w, vi + 0.004, _pct(vi, 1), ha="center", fontsize=7.2, color=TESTO2)
    ax.set_xticks(x, [f"Case {k}\nΩ = {d.loc[k, 'Elasticità Ω = Δ·E/V']:.2f}" for k in "ABC"])
    _asse_pct(ax)
    ax.set_ylim(0, 0.28)
    ax.set_ylabel("Discount rate")
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    _legenda_sotto(fig, h, l, 3)
    _salva(fig, "Fig12_discount_rate")


def fig13a_pwerm_tasso(ts):
    d = ts[ts.Data == "valutazione"].set_index("Caso")
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 2.6), sharey=True)
    ymax = d["PWERM da quantili (limite) al Ke (€m)"].max() * 1.15
    for ax, k in zip(assi, "ABC"):
        opm = d.loc[k, "Valore OPM (€m)"]
        v = [d.loc[k, "PWERM da quantili (limite) al Ke (€m)"], d.loc[k, "PWERM da quantili (limite) al rendimento del MIP (€m)"]]
        ax.bar([0, 1], v, 0.6, color=[SERIE[0], SERIE[1]], zorder=2)
        ax.axhline(opm, color=TESTO, lw=1, ls="--", zorder=3)
        for xi, vi in enumerate(v):
            ax.annotate(f"{vi:.1f}  ({vi / opm - 1:+.0%})", (xi, vi), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=TESTO2, zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_xticks([0, 1], ["at Ke", "at MIP\nrequired return"], fontsize=7.7)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, ymax)
        ax.set_title(f"Case {k} – OPM {opm:.1f}", loc="left")
    assi[0].set_ylabel("MIP value (€m)")
    h = [Patch(color=SERIE[0]), Patch(color=SERIE[1]), Line2D([], [], color=TESTO, lw=1, ls="--")]
    _legenda_sotto(fig, h, ["PWERM discounted at Ke", "PWERM discounted at rf + Ω(Ke − rf)", "OPM value"], 3)
    _salva(fig, "Fig13a_pwerm_full_distribution")


def fig13b_pwerm_ponte(ts):
    d = ts[ts.Data == "valutazione"].set_index("Caso")
    fig, assi = plt.subplots(1, 3, figsize=(LARGHEZZA, 2.9), sharey=True)
    ymax = max(d["Valore OPM (€m)"].max(), d["PWERM da prassi al Ke (€m)"].max()) * 1.15
    etichette = ["OPM", "Missing\ntail", "Ke\nrate", "Practice\nPWERM"]
    for ax, k in zip(assi, "ABC"):
        opm = d.loc[k, "Valore OPM (€m)"]
        p_mip = d.loc[k, "PWERM da prassi al rendimento del MIP (€m)"]
        p_ke = d.loc[k, "PWERM da prassi al Ke (€m)"]
        passi = [(0, opm, TESTO2), (opm, p_mip - opm, SERIE[1]), (p_mip, p_ke - p_mip, SERIE[0]), (0, p_ke, TESTO2)]
        for i, (base, delta, colore) in enumerate(passi):
            ax.bar(i, delta, 0.62, bottom=base, color=colore)
            top = base + max(delta, 0)
            testo = f"{delta:.1f}" if i in (0, 3) else f"{delta:+.1f}"
            ax.annotate(testo, (i, top), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=TESTO)
        for i in range(3):
            livello = [opm, p_mip, p_ke][i]
            ax.plot([i + 0.31, i + 0.69], [livello, livello], color=NEUTRO, lw=0.8)
        ax.set_xticks(range(4), etichette, fontsize=7.5)
        ax.set_xlim(-0.55, 3.55)
        ax.set_title(f"Case {k}", loc="left")
        ax.set_ylim(0, ymax)
        ax.grid(axis="x", visible=False)
    assi[0].set_ylabel("MIP value (€m)")
    h = [Patch(color=SERIE[1]), Patch(color=SERIE[0])]
    _legenda_sotto(fig, h, ["value lost by using six scenarios instead of the full distribution",
                            "value added by discounting at Ke (lower rate)"], 1)
    _salva(fig, "Fig13b_pwerm_practice_bridge")


def tabella_conflitto(cf):
    righe = []
    for k, soglia in (("A", "ratchet"), ("B", "fine del catch-up")):
        d = cf[(cf.Caso == k) & (cf.Soglia == soglia)]
        finestre = list(dict.fromkeys(d.Finestre))
        for regola in ("Regola obiettivi", "LSM (ottimo del fondo)"):
            x = d[(d.Regola == regola) & (d.Finestre == finestre[0])]
            if regola.startswith("LSM") and str(x.Politica.iloc[0]).startswith("Exit fissa") and len(finestre) > 1:
                x = d[(d.Regola == regola) & (d.Finestre == finestre[1])]
            r = x.iloc[0]
            nome = "Target rule" if "obiettivi" in regola else "LSM (fund-optimal)"
            if r.Finestre != finestre[0]:
                nome += f", windows {r.Finestre}"
            righe.append({"Case": k, "Threshold": {"ratchet": "ratchet", "fine del catch-up": "end of catch-up"}[soglia],
                          "Exit rule": nome, "Exit windows": r.Finestre,
                          "P(exit | equity 90–100% of threshold) (%)": r["P(exit | equity tra 90% e 100% della soglia)"],
                          "P(exit | equity 100–110% of threshold) (%)": r["P(exit | equity tra 100% e 110% della soglia)"]})
    return pd.DataFrame(righe)


def fig14_conflitto(t):
    fig, assi = plt.subplots(1, 2, figsize=(LARGHEZZA, 2.6), sharey=True)
    titoli = {"A": "Case A – ratchet threshold", "B": "Case B – end of catch-up"}
    c1, c2 = "P(exit | equity 90–100% of threshold) (%)", "P(exit | equity 100–110% of threshold) (%)"
    for ax, k in zip(assi, "AB"):
        d = t[t.Case == k]
        x = np.arange(len(d))
        ax.bar(x - 0.18, d[c1], 0.34, color=SERIE[0], label="equity just below the threshold (90–100%)")
        ax.bar(x + 0.18, d[c2], 0.34, color=SERIE[1], label="equity just above the threshold (100–110%)")
        for xi, (a, b) in enumerate(zip(d[c1], d[c2])):
            ax.text(xi - 0.18, a + 0.02, _pct(a), ha="center", fontsize=7.7, color=TESTO2)
            ax.text(xi + 0.18, b + 0.02, _pct(b), ha="center", fontsize=7.7, color=TESTO2)
        ax.set_xticks(x, [r.replace(", windows", "\nwindows") for r in d["Exit rule"]])
        ax.set_title(titoli[k], loc="left")
        _asse_pct(ax)
        ax.set_ylim(0, 1.12)
    assi[0].set_ylabel("Probability that the fund exits")
    fig.tight_layout()
    h, l = assi[0].get_legend_handles_labels()
    _legenda_sotto(fig, h, l, 2)
    _salva(fig, "Fig14_exit_conflict")


def fig15_strutturale(cal, scomp):
    c = cal[(cal.Data == "valutazione") & (cal["σ attivo"] == "peer")].set_index("Caso")
    s = scomp[(scomp.Data == "valutazione") & (scomp.Configurazione == "Exit incerta, con leaver")].set_index("Caso")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARGHEZZA, 2.7))
    x = np.arange(3)
    w = 0.36
    for i, (col, colore, et) in enumerate((("Volatilità equity del GBM", TESTO2,
                                            "input of OPM, trees and MC (relevered at today's leverage)"),
                                           ("Volatilità equity implicita iniziale", SERIE[0],
                                            "implied by the structural model (fixed interest and repayments)"))):
        v = c.loc[list("ABC"), col].values
        a1.bar(x + (i - 0.5) * w, v, w * 0.92, color=colore, label=et)
        for xi, vi in zip(x, v):
            a1.text(xi + (i - 0.5) * w, vi + 0.008, _pct(vi), ha="center", fontsize=7.5, color=TESTO2)
    a1.set_xticks(x, [f"Case {k}" for k in "ABC"])
    a1.set_ylim(0, 0.62)
    _asse_pct(a1)
    a1.set_title("Equity volatility at the valuation date", loc="left", fontsize=9.4)
    a1.grid(axis="x", visible=False)
    mc = s.loc[list("ABC"), "MC sull'equity (€m)"].values
    comp = [("  di cui volatilità di partenza (σ peer − σ coerente)", SERIE[1], "starting volatility (structural vs relevered)"),
            ("  di cui dinamica di leva (σ coerente − GBM)", SERIE[3], "falling leverage over time")]
    for i, (col, colore, et) in enumerate(comp):
        v = s.loc[list("ABC"), col].values / mc
        a2.bar(x + (i - 0.5) * w, v, w * 0.92, color=colore, label=et)
        for xi, vi in zip(x, v):
            a2.text(xi + (i - 0.5) * w, vi + (0.008 if vi >= 0 else -0.008), f"{vi * 100:+.0f}%", ha="center",
                    va="bottom" if vi >= 0 else "top", fontsize=7.5, color=TESTO2)
    a2.axhline(0, color=TESTO, lw=1)
    a2.set_xticks(x, [f"Case {k}" for k in "ABC"])
    a2.set_ylim(-0.33, 0.1)
    _asse_pct(a2)
    a2.set_title("Structural vs full Monte Carlo value", loc="left", fontsize=9.4)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    _legenda_sotto(fig, h1 + h2, ["equity volatility: " + l for l in l1] + ["value effect: " + l for l in l2], 1)
    _salva(fig, "Fig15_structural_model")


INDICE = [
    ("Fig01", "Payoff profile", "MIP pool payoff at exit as a function of the equity value, at the business-plan exit date. Vertical lines mark the hurdle and the points where the slope changes (ratchet in Case A, end of the catch-up in Case B).",
     "Shows the contractual shape that every model has to value."),
    ("Fig02a", "Payoff distribution – probability", "Probability that the MIP payoff at exit falls in each bucket of the multiple of the price paid by management (real-world measure, exit at the business-plan date).",
     "Shows that the MIP is most likely worthless."),
    ("Fig02b", "Payoff distribution – value share", "Share of the expected MIP payoff coming from each bucket of the multiple of the price paid by management (real-world measure, exit at the business-plan date).",
     "Shows that the expected value of the MIP comes from the upper tail."),
    ("Fig03", "Values by model", "MIP value at the valuation date by model (€m). Dotted line: full Monte Carlo. The quantile PWERM is a diagnostic benchmark (real-world distribution discounted at Ke), not a valuation candidate. The endogenous-exit models are not applied to Case C in the base design.",
     "Model risk expressed in euros."),
    ("Fig04", "Dispersion by group", "Dispersion across models, (max − min) / median, as model families are added to the comparison. Hatched bars: the group mixes models with different exit or leaver assumptions (practice PWERM, endogenous exit, quantile PWERM), so the dispersion is indicative only – it is not a like-for-like comparison of models.",
     "Separates numerical convergence from model risk proper."),
    ("Fig05", "Numerical convergence", "Accuracy of the numerical methods (fixed exit, no leavers), on a common logarithmic scale. Top: pricing error of the binomial and trinomial trees against the closed-form Black-Scholes value. Bottom: standard error of the Monte Carlo estimate, plain and with a control variate (a call option on the equity value with strike equal to the hurdle, whose exact value is known). In Case C the MIP payoff coincides with the control variate, so the corrected estimate has zero error.",
     "Shows that implementation risk is negligible."),
    ("Fig06", "Art. 60 by model", "Art. 60 test at entry: MIP fair value / total investment at entry by model, against the 1% threshold.",
     "Shows that in Case B the outcome of the test depends on the model."),
    ("Fig06b", "Art. 60 headroom spread", "Art. 60 headroom by case: percentage fall in the MIP fair value (equivalently, in the size of the MIP pool) that would bring it to 1% of the total investment at entry. Each dot is a valuation model (the diagnostic quantile PWERM is excluded); the shaded bar spans the lowest and the highest headroom; the figure on the right is the spread in percentage points.",
     "Shows how much the choice of model moves the margin on the 1% test."),
    ("Fig07", "Parameter vs model risk", "Range of the value effect of each estimation parameter (plausible interval, full Monte Carlo), compared with the dispersion across models.",
     "Puts parameter risk and model risk on the same scale."),
    ("Fig08", "Tornado", "Minimum and maximum change in the full Monte Carlo value for each estimation parameter (plausible interval).",
     "Direction and size of parameter risk."),
    ("Fig09", "Art. 60 sensitivity (Case B)", "Case B: MIP fair value / total investment at entry for each plausible parameter test, simple OPM vs full Monte Carlo. Below the dashed line: volatility convention test, with the median equity volatility of the peers used without relevering to the leverage of the deal (a methodological choice, not an estimation interval). The leverage definition is not shown: at entry there are no debt-like items, so it leaves the value unchanged. Volatility estimation window at entry: 1 and 2 years ending 31 December 2025, against the base window 2019–2025 without the Covid-19 periods.",
     "Robustness of the 1% test to estimation parameters and to the volatility convention."),
    ("Fig10", "Conditional map – thresholds", "Dispersion between simple OPM, full Monte Carlo and the structural model, by equity volatility and level of the waterfall thresholds. The hurdle level in equity value at the base exit date is multiplied by 0.8, 1 or 1.2 (money-multiple and IRR hurdles alike); the ratchet of Case A moves by the same factor and the end of the catch-up of Case B follows the hurdle (about ±26%). Darker cells: larger dispersion. Framed cell: base case.",
     "Shows under which conditions the models diverge most."),
    ("Fig11", "Conditional map – exit", "Dispersion between simple OPM and full Monte Carlo, by equity volatility and exit date. Darker cells: larger dispersion. Framed cell: base case.",
     "Shows how stable the cost of the simplifying assumptions is."),
    ("Fig12", "Discount rate", "Equity cost of capital (Ke) versus the return required by the MIP, rf + Ω(Ke − rf), and the rate implied by the option model (valuation date). Ω is the elasticity of the MIP value to the equity value.",
     "Shows that the MIP must be discounted at a higher rate than Ke."),
    ("Fig13a", "PWERM – full distribution", "PWERM with the full distribution of exit values: difference from the OPM value when discounted at Ke and at the MIP required return (valuation date).",
     "Shows that a PWERM must be discounted at the MIP required return, not at Ke."),
    ("Fig13b", "PWERM – practice scenarios", "PWERM with the six practice scenarios: bridge from the OPM value to the practice PWERM discounted at Ke. The scenarios omit the upper tail of the distribution and lose value; discounting at Ke instead of the MIP required return adds value back. In Cases A and B the two errors largely offset; in Case C they do not.",
     "Shows that practice scenarios discounted at Ke look right only because two errors offset."),
    ("Fig14", "Exit conflict", "Probability that the fund exits when the equity value is just below versus just above a management threshold (risk-neutral measure). Target rule: the fund exits at the first annual window in which its own net IRR and money multiple targets are met. LSM: the fund chooses the exit date that maximises its own value (Longstaff-Schwartz regression). Exit windows as in the exogenous distribution (Case A 2029–2033, Case B 2028–2034).",
     "Shows the fund's incentive to exit before the ratchet triggers or the catch-up ends."),
    ("Fig15", "Structural model", "Left: equity volatility at the valuation date – the input of the equity-based models (peer asset volatility relevered at today's leverage) and the volatility implied by the structural model, where interest and repayments are fixed cash flows taken from the business plan. Right: difference between the structural and the full Monte Carlo value, split into the effect of the starting volatility and the effect of the falling leverage over time.",
     "Separates the effect of the starting volatility from that of the falling leverage."),
]


def tabelle(valori, gruppi, art60, mvp, distr, fasce, ts, g3, g2e, conflitto):
    v = valori.assign(Model=valori.Modello.map(_modello), Family=valori.Famiglia.map(FAMIGLIE))
    t1 = v.pivot_table(index=["Model", "Family"], columns="Caso", values="Valore alla valutazione (€m)", sort=False)
    t1.columns = [f"Case {c} (€m)" for c in t1.columns]
    t1 = t1.reset_index()

    t2 = pd.DataFrame({"Case": gruppi.Caso, "Model group": gruppi["Gruppo di modelli"].str[0].map(GRUPPI),
                       "Homogeneous assumptions": gruppi["Ipotesi omogenee"].map(SI_NO),
                       "Min (€m)": gruppi["Minimo (€m)"], "Max (€m)": gruppi["Massimo (€m)"],
                       "Median (€m)": gruppi["Mediana (€m)"], "Dispersion (%)": gruppi.Dispersione,
                       "Lowest model": gruppi["Modello minimo"].map(_modello),
                       "Highest model": gruppi["Modello massimo"].map(_modello)})

    a = art60.assign(Model=art60.Modello.map(_modello))
    t3 = a.pivot_table(index="Model", columns="Caso", values="Fair value / equity", sort=False)
    t3.columns = [f"Case {c} – FV / total investment (%)" for c in t3.columns]
    t3 = t3.reset_index().merge(tabella_margine(art60), on="Model")
    t3b = tabella_ampiezza_margine(art60)

    t4 = pd.DataFrame({"Case": mvp.Caso,
                       "Model risk, group 2 (%)": mvp["Model risk: OPM semplice vs modelli completi (gruppo 2)"],
                       "Model risk, group 3 (%)": mvp["Model risk: + strutturale (gruppo 3)"],
                       "Most influential estimation parameter": mvp["Parametro di stima più influente"].map(PARAMETRI),
                       "Its range, plausible interval (%)": mvp["Sua ampiezza (intervallo plausibile)"],
                       "Parameters with range > group 2": mvp["Parametri di stima con ampiezza > gruppo 2"],
                       "Parameters with range > group 3": mvp["Parametri di stima con ampiezza > gruppo 3"],
                       "Estimation parameters tested": mvp["Parametri di stima valutati"]})

    t6 = pd.DataFrame({"Date": ts.Data.map(DATE), "Case": ts.Caso, "OPM value (€m)": ts["Valore OPM (€m)"],
                       "Delta (Δ)": ts["Delta (Δ)"], "Elasticity Ω = Δ·E/V": ts["Elasticità Ω = Δ·E/V"],
                       "Ke (%)": ts["Ke (equity)"], "Risk-free rate (%)": ts["Risk-free annuo"],
                       "MIP required return rf + Ω(Ke − rf) (%)": ts["Rendimento richiesto dal MIP (CAPM con Ω)"],
                       "Rate implied by the model (%)": ts["Tasso implicito (distribuzione del modello)"],
                       "Rate aligning practice scenarios to OPM (%)": ts["Tasso implicito (scenari di prassi)"],
                       "PWERM full distribution at Ke (€m)": ts["PWERM da quantili (limite) al Ke (€m)"],
                       "PWERM full distribution at MIP return (€m)": ts["PWERM da quantili (limite) al rendimento del MIP (€m)"],
                       "PWERM practice scenarios at Ke (€m)": ts["PWERM da prassi al Ke (€m)"],
                       "PWERM practice scenarios at MIP return (€m)": ts["PWERM da prassi al rendimento del MIP (€m)"]})

    def _griglia(g, prefisso, nuovo):
        out = g.rename(columns={"Caso": "Case", "Δ volatilità": "Change in equity volatility (%)"})
        return out.rename(columns={c: c.replace(prefisso, nuovo) + " (%)" for c in out.columns if c.startswith(prefisso)})

    t7a = _griglia(g3, "Fattore soglie =", "Threshold multiplier ×")
    t7b = _griglia(g2e, "Δ exit (anni) =", "Exit shift (years)")
    return {"T1_Values_by_model": t1, "T2_Dispersion_by_group": t2, "T3_Art60_by_model": t3, "T3b_Art60_headroom_spread": t3b,
            "T4_Parameter_vs_model_risk": t4, "T5_Payoff_distribution": distr, "T5b_Payoff_buckets": fasce,
            "T6_Discount_rate": t6, "T7a_Map_vol_x_thresholds": t7a, "T7b_Map_vol_x_exit": t7b,
            "T8_Exit_conflict": conflitto}


def _formato(colonna: str, valori: pd.Series):
    c = colonna.lower()
    if "(%)" in colonna:
        return "0.00%" if ("equity" in c and "volatility" not in c) else "0.0%"
    if "(€m)" in colonna:
        return "#,##0.00"
    if "(x price)" in colonna:
        return '0.00"x"'
    if pd.api.types.is_float_dtype(valori):
        return "0.000"
    return None


def salva_excel(tab: dict[str, pd.DataFrame], percorso=FILE_TABELLE):
    info = pd.DataFrame({"Item": ["Run date", "Input file", "Input file fingerprint"],
                         "Value": [f"{dt.datetime.now():%d/%m/%Y %H:%M}", FILE_INPUT.name, hash_file(FILE_INPUT)]})
    tutte = {"Info": info, **tab}
    intestazione = PatternFill("solid", fgColor="1F3A5F")
    bordo = Border(bottom=Side(style="thin", color="BFBFBF"))
    with pd.ExcelWriter(percorso, engine="openpyxl") as xw:
        for nome, df in tutte.items():
            df.to_excel(xw, sheet_name=nome[:31], index=False)
            ws = xw.sheets[nome[:31]]
            for j, col in enumerate(df.columns, start=1):
                h = ws.cell(1, j)
                h.font = Font(bold=True, color="FFFFFF")
                h.fill = intestazione
                h.alignment = Alignment(wrap_text=True, vertical="center")
                f = _formato(str(col), df[col])
                testo = df[col].astype(str)
                larghezza = max(10, min(60 if df[col].dtype == object else 16, int(testo.str.len().max() or 0) + 2))
                if df[col].dtype != object:
                    larghezza = max(12, min(18, len(str(col)) // 2 + 2))
                ws.column_dimensions[get_column_letter(j)].width = larghezza
                for i in range(2, len(df) + 2):
                    cella = ws.cell(i, j)
                    cella.border = bordo
                    if f:
                        cella.number_format = f
                    if df[col].dtype == object:
                        cella.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[1].height = 45
            ws.freeze_panes = "A2"


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    valori = _leggi("07_confronto_modelli.xlsx", "Valori_per_modello")
    gruppi = _leggi("07_confronto_modelli.xlsx", "Dispersione_gruppi")
    art60 = _leggi("07_confronto_modelli.xlsx", "Art60_per_modello")
    alberi = _leggi("03_alberi.xlsx", "Convergenza_alberi")
    mc = _leggi("04_montecarlo.xlsx", "Convergenza_MC")
    conflitto = _leggi("06_exit_endogena.xlsx", "Conflitto_soglie")
    cal = _leggi("05_strutturale.xlsx", "Calibrazione")
    scomp = _leggi("05_strutturale.xlsx", "Scomposizione")
    tornado = _leggi("08_sensitivity.xlsx", "Tornado_stima_plausibile")
    mvp = _leggi("08_sensitivity.xlsx", "Model_vs_parameter_risk")
    sens = _leggi("08_sensitivity.xlsx", "Sensitivity")
    convenzione = _leggi("08_sensitivity.xlsx", "Convenzione_volatilita")
    g3 = _leggi("09_approfondimenti.xlsx", "Griglia_vol_soglie_g3")
    g2e = _leggi("09_approfondimenti.xlsx", "Griglia_vol_exit_g2")
    ts = tabella_tasso_sconto(casi)

    profilo = fig01_profilo(casi)
    distr, fasce = tabella_distribuzione(casi)
    fig02_distribuzione(fasce)
    fig03_valori(valori)
    fig04_gruppi(gruppi)
    conv = fig05_convergenza(alberi, mc)
    fig06_art60(art60)
    fig06b_margine(art60)
    fig07_parametri_vs_modello(tornado, mvp, valori, _leggi("04_montecarlo.xlsx", "Leaver_fair_value"))
    fig08_tornado(tornado)
    a60s = fig09_art60_sensitivity(sens)
    fig10_mappa_g3(g3)
    fig11_mappa_exit(g2e)
    fig12_tassi(ts)
    fig13a_pwerm_tasso(ts)
    fig13b_pwerm_ponte(ts)
    t_confl = tabella_conflitto(conflitto)
    fig14_conflitto(t_confl)
    fig15_strutturale(cal, scomp)

    indice = pd.DataFrame(INDICE, columns=["Figure", "Short title", "Suggested caption", "Purpose"])
    indice["File"] = [f"grafici/{f}_*.png" for f in indice.Figure]
    sc = scomp[(scomp.Data == "valutazione") & (scomp.Configurazione == "Exit incerta, con leaver")].set_index("Caso")
    cv = cal[(cal.Data == "valutazione") & (cal["σ attivo"] == "peer")].set_index("Caso")
    t9 = pd.DataFrame({"Case": list("ABC"),
                       "Equity volatility, input of equity-based models (%)": cv.loc[list("ABC"), "Volatilità equity del GBM"].values,
                       "Equity volatility implied by the structural model (%)": cv.loc[list("ABC"), "Volatilità equity implicita iniziale"].values,
                       "Peer asset volatility (%)": cv.loc[list("ABC"), "Volatilità dell'attivo"].values,
                       "Full Monte Carlo (€m)": sc.loc[list("ABC"), "MC sull'equity (€m)"].values,
                       "Structural, peer σ (€m)": sc.loc[list("ABC"), "Strutturale, σ peer (€m)"].values,
                       "Effect of starting volatility (€m)": sc.loc[list("ABC"), "  di cui volatilità di partenza (σ peer − σ coerente)"].values,
                       "Effect of leverage dynamics (€m)": sc.loc[list("ABC"), "  di cui dinamica di leva (σ coerente − GBM)"].values})
    return {"Index": indice, "T9_Structural_vs_MC": t9, **tabelle(valori, gruppi, art60, mvp, distr, fasce, ts, g3, g2e, t_confl),
            "T10_Volatility_convention": tabella_convenzione(convenzione),
            "Data_Fig01": profilo, "Data_Fig05": conv, "Data_Fig09": a60s}
