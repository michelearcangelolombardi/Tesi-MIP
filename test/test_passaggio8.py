import pytest

from mip.input import carica_casi
from mip.montecarlo import valore_mc
from mip.opm import mercato, valore_opm
from mip.sensitivity import MC, OPM, caso_leva, griglia, prove, valori


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_griglie_dal_foglio():
    assert griglia("Volatilità equity") == [-0.1, -0.05, 0.0, 0.05, 0.1]
    assert griglia("Probabilità annua di uscita") == [0.0, 0.025, 0.05, 0.075, 0.1]
    from mip.sensitivity import LEAVER_EXTRA, STRESS_EXIT
    assert max(LEAVER_EXTRA) == 0.20 and STRESS_EXIT == (-2.0, 2.0)


def test_livello_base_riproduce_il_caso_base(casi):
    for k, c in casi.items():
        base = {OPM: valore_opm(c, mercato(c)), MC: valore_mc(c, exit_incerta=True, con_leaver=True).valore}
        for p in prove(c):
            if p.base:
                v = valori(p, "valutazione", 5)
                for m in (OPM, MC):
                    if m in v:
                        assert abs(v[m] - base[m]) < 1e-9, (k, p.parametro, m)


def test_quote_lineari_e_dlom(casi):
    for k, c in casi.items():
        base = valore_opm(c, mercato(c))
        for p in prove(c):
            if p.parametro == "Quote del management":
                assert abs(valori(p, "valutazione", 5)[OPM] / base - p.numero) < 1e-9, k
            if p.parametro == "DLOM":
                assert abs(valori(p, "valutazione", 5)[OPM] / base - (1 - p.numero)) < 1e-12, k


def test_leva_regola_25(casi):
    for k, c in casi.items():
        x = caso_leva(c, 0.05)
        assert x.equity_ingresso < c.equity_ingresso and x.vol_valutazione > c.vol_valutazione
        assert abs(x.investimento_fondo - (x.equity_ingresso - c.importo_mip)) < 1e-9
