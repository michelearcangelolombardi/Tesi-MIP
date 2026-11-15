import pytest

from mip.input import carica_casi
from mip.passaggio9 import tabella_mappa, tabella_tasso_sconto


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_elasticita_e_tassi(casi):
    t = tabella_tasso_sconto(casi)
    assert (t["Elasticità Ω = Δ·E/V"] > 1).all()
    assert (t["Tasso implicito (distribuzione del modello)"] > t["Ke (equity)"]).all()
    diff = (t["Rendimento richiesto dal MIP (CAPM con Ω)"] - t["Tasso implicito (distribuzione del modello)"]).abs()
    assert (diff < 0.02).all()


def test_mappa_base_coerente_col_passaggio7(casi):
    m = tabella_mappa({"B": casi["B"]})
    b = m[(m.Mappa == "Volatilità × soglie") & (m["Δ volatilità"] == 0) & (m["Fattore soglie"] == 1.0)].iloc[0]
    assert 0.10 < b["Dispersione gruppo 2 (OPM vs MC completo)"] < 0.25
    lo = m[(m.Mappa == "Volatilità × soglie") & (m["Δ volatilità"] == -0.10) & (m["Fattore soglie"] == 1.2)].iloc[0]
    assert lo["Dispersione gruppo 3 (+ strutturale)"] > b["Dispersione gruppo 3 (+ strutturale)"]
