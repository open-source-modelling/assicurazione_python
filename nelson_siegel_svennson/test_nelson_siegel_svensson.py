import numpy as np
import pytest
import nelsonsiegelsvensson
from nelsonsiegelsvensson import NelsonSiegelSvensson, NSSMinimize

TimeVec = np.array([1, 2, 5, 10, 25])
YieldVec = np.array([0.0039, 0.0061, 0.0166, 0.0258, 0.0332])

# Esempio del README e di main.py
def test_esempio_readme_adatta_i_dati():
    params = NSSMinimize(0.1, 0.1, 0.1, 0.1, 1, 1, TimeVec, YieldVec)
    assert params[4] > 0 and params[5] > 0
    assert np.allclose(NelsonSiegelSvensson(TimeVec, *params), YieldVec, atol=1e-6)

# Per T = 0 la curva è il limite beta0 + beta1 invece di nan
def test_scadenza_zero():
    params = (0.04, -0.03, -0.05, -0.01, 1.3, 5.6)
    out = NelsonSiegelSvensson(np.array([0.0, 1e-9]), *params)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(params[0] + params[1])
    assert out[0] == pytest.approx(out[1])

# NSSMinimize restituiva una lista vuota, quindi il chiamante falliva più avanti con un IndexError
def test_errore_se_ottimizzazione_fallisce(monkeypatch):
    class Fallita:
        success = False
        message = "Maximum number of function evaluations has been exceeded."
    monkeypatch.setattr(nelsonsiegelsvensson, "minimize", lambda *args, **kwargs: Fallita())
    with pytest.raises(RuntimeError, match="non è convergente"):
        NSSMinimize(0.1, 0.1, 0.1, 0.1, 1, 1, TimeVec, YieldVec)
