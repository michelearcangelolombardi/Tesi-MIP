import math

import numpy as np
import pytest

from mip.input import carica_casi
from mip.opm import mercato, valore_opm
from mip.passaggio10 import tabella_distribuzione


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_media_risk_neutral_scontata_uguale_a_opm(casi):
    t, _ = tabella_distribuzione(casi)
    for k, c in casi.items():
        r = t[(t.Case == k) & t.Measure.str.startswith("Risk-neutral")].iloc[0]
        v = r["Mean (x price)"] * c.importo_mip * math.exp(-c.rf_cont_valutazione * c.T)
        assert abs(v / valore_opm(c, mercato(c)) - 1) < 0.03, k


def test_probabilita_coerenti(casi):
    t, _ = tabella_distribuzione(casi)
    assert ((t["P(MIP = 0) (%)"] <= t["P(payoff < price paid) (%)"]) & (t["P(MIP = 0) (%)"] > 0)).all()
    for k in casi:
        p = t[t.Case == k].set_index("Measure")["P(MIP = 0) (%)"]
        assert p.iloc[0] < p.iloc[1]


def test_fasce_sommano_a_uno(casi):
    _, f = tabella_distribuzione(casi)
    for k in casi:
        d = f[f.Case == k]
        assert abs(d["Probability (%)"].sum() - 1) < 1e-12
        assert abs(d["Share of expected payoff (%)"].sum() - 1) < 1e-12
