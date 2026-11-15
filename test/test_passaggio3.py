import pytest

from mip.alberi import valore_albero, valore_opm_exit_incerta
from mip.input import carica_casi
from mip.opm import mercato, valore_opm


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_alberi_convergono_a_black_scholes(casi):
    for k, c in casi.items():
        bs = valore_opm(c, mercato(c))
        bin_ = (valore_albero(c, tipo="binomiale", N=1000) + valore_albero(c, tipo="binomiale", N=1001)) / 2
        tri = valore_albero(c, tipo="trinomiale", N=2000)
        assert abs(bin_ / bs - 1) < 0.002, (k, bin_, bs)
        assert abs(tri / bs - 1) < 0.005, (k, tri, bs)


def test_exit_incerta_uguale_alla_media_delle_opm(casi):
    for k, c in casi.items():
        atteso = valore_opm_exit_incerta(c)
        for tipo in ("binomiale", "trinomiale"):
            v = valore_albero(c, tipo=tipo, exit_incerta=True)
            assert abs(v / atteso - 1) < 0.005, (k, tipo, v, atteso)


def test_leaver_con_probabilita_zero_non_cambia_il_valore(casi):
    for k, c in casi.items():
        senza = valore_albero(c, tipo="binomiale", N=200)
        zero = valore_albero(c, tipo="binomiale", N=200, con_leaver=True, prob_leaver=0.0)
        assert abs(senza - zero) < 1e-12, k


def test_leaver_riduce_il_valore(casi):
    for k, c in casi.items():
        v = [valore_albero(c, tipo="binomiale", N=200, con_leaver=True, prob_leaver=p) for p in (0.0, 0.025, 0.05, 0.10)]
        assert all(a > b for a, b in zip(v[:-1], v[1:])), (k, v)
