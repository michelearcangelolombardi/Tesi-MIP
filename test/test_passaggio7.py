import pytest

from mip.input import carica_casi
from mip.montecarlo import valore_mc
from mip.passaggio7 import tabella_gruppi, valori_modelli


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


@pytest.fixture(scope="module")
def valori(casi):
    return {k: valori_modelli(c, "valutazione") for k, c in casi.items()}


def test_valori_coerenti_con_i_passaggi_precedenti(casi, valori):
    for k, c in casi.items():
        assert valori[k]["Monte Carlo completo"] == valore_mc(c, exit_incerta=True, con_leaver=True).valore


def test_motori_diversi_convergono(valori):
    g = tabella_gruppi(valori)
    assert (g[g["Gruppo di modelli"].str.startswith("1.")]["Dispersione"] < 0.02).all()


def test_gruppi_annidati(valori):
    g = tabella_gruppi(valori)
    for _, x in g.groupby("Caso"):
        d = (x["Massimo (€m)"] - x["Minimo (€m)"]).tolist()
        assert all(a <= b + 1e-12 for a, b in zip(d[:-1], d[1:]))
