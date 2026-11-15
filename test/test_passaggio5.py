import pytest

from mip.input import carica_casi
from mip.montecarlo import valore_mc
from mip.strutturale import calibra, sigma_attivo_coerente, verifica_calibrazione_mc

DATE = ("valutazione", "ingresso")


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_v0_riproduce_l_equity_osservato(casi):
    for k, c in casi.items():
        for d in DATE:
            cal = calibra(c, d)
            media, se = verifica_calibrazione_mc(c, d)
            assert abs(media - cal.E0) < 3 * se, (k, d, media, cal.E0, se)


def test_sigma_attivo_coerente(casi):
    for k, c in casi.items():
        for d in DATE:
            cal = calibra(c, d, sigma_attivo_coerente(c, d))
            assert abs(cal.vol_equity_implicita - cal.vol_equity_gbm) < 1e-6, (k, d)


def test_mc_strutturale_con_e_senza_controllo(casi):
    for k, c in casi.items():
        a = valore_mc(c, modello="strutturale", controllo=True)
        b = valore_mc(c, modello="strutturale", controllo=False)
        assert abs(a.valore - b.valore) < 3 * b.errore_standard, (k, a.valore, b.valore)
