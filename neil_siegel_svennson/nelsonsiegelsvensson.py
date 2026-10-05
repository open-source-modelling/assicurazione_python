import numpy as np
from scipy.optimize import minimize

def NelsonSiegelSvensson(T, beta0, beta1, beta2, beta3, lambda0, lambda1):
    """
    NelsonSiegelSvensson calcola la curva interpolata/estrappolata nei punti dell'array "T" utilizzando l'algoritmo di Nelson-Siegel-Svensson (NSS),
    parametrizzato con i parametri beta0, beta1, beta2, beta3, lambda0, lambda1. Restituisce un ndarray numpy di punti.
    
    Argomenti:
        T: ndarray n x 1 delle scadenze per le quali si desidera calcolare il tasso corrispondente.
        beta0: floating 1 x 1, rappresentanta il primo fattore della parametrizzazione NSS.
        beta1: floating 1 x 1, rappresentanta il secondo fattore della parametrizzazione NSS.
        beta2: floating 1 x 1, rappresentanta il terzo fattore della parametrizzazione NSS.
        beta3: floating 1 x 1, rappresentanta il quarto fattore della parametrizzazione NSS.
        lambda0: floating 1 x 1, rappresentanta il primo parametro di forma lambda della parametrizzazione NSS.
        lambda1: floating 1 x 1, rappresentanta il secondo parametro di forma lambda della parametrizzazione NSS.
        
    Restituisce:
        ndarray n x 1 di punti interpolati/estrappolati corrispondenti alle scadenze all'interno di T. Dove n è la lunghezza del vettore T.
        Per T = 0 il tasso è il limite beta0 + beta1.

    Implementato da Gregor Fabjan di Qnity Consultants il 16/11/2023
    """
    T = np.asarray(T, dtype=float)

    def fattore_pendenza(lam):
        # (1 - exp(-T/lam)) / (T/lam), che tende a 1 quando T tende a 0
        x = T / lam
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x == 0, 1.0, (1 - np.exp(-x)) / x)

    alpha1 = fattore_pendenza(lambda0)
    alpha2 = alpha1 - np.exp(-T / lambda0)
    alpha3 = fattore_pendenza(lambda1) - np.exp(-T / lambda1)

    return beta0 + beta1 * alpha1 + beta2 * alpha2 + beta3 * alpha3


def NSSGoodFit(params, TimeVec, YieldVec):
    """
    NSSGoodFit calcola i residui tra il rendimenti osservati nel mercato e quelli previsti dall'algoritmo NSS con la parametrizzazione specificata.
    
    Argomenti:
        params: tuple 6 x 1 continene i 6 parametri dell'algoritmo NSS. La sequenza dei parametri deve essere (beta0, ..., beta3, lambda0, lambda1).
        TimeVec: ndarray n x 1 di scadenze per cui sono stati osservati i rendimenti in YieldVec.
        YieldVec: ndarray n x 1 di rendimenti osservati.

    Restituisce:
        float 1 x 1, somma dei quadrati delle differenze tra i punti calcolati e i dati osservati.
        
    Implementato da Gregor Fabjan di Qnity Consultants il  16/11/2023
    """

    return np.sum((NelsonSiegelSvensson(TimeVec, params[0], params[1], params[2], params[3], params[4], params[5])-YieldVec)**2)

def NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, TimeVec, YieldVec):
    """
    NSSMinimize utilizza la funzione di minimizzazione incorporata nella libreria scipy di Python. La funzione configura i parametri e la funzione NSSGoodFit in modo
    che sia compatibile con il modo in cui la funzione minimize richiede i suoi argomenti. I parametri di forma lambda0 e lambda1 sono mantenuti positivi. Se l'ottimizzazione non converge, viene sollevato un RuntimeError.
    
    Argomenti:
        beta0: numero decimale 1 x 1, rappresentanta il primo fattore della parametrizzazione NSS.
        beta1: numero decimale 1 x 1, rappresentanta il secondo fattore della parametrizzazione NSS.
        beta2: numero decimale 1 x 1, rappresentanta il terzo fattore della parametrizzazione NSS.
        beta3: numero decimale 1 x 1, rappresentanta il quarto fattore della parametrizzazione NSS.
        lambda0: numero decimale 1 x 1, rappresentanta il primo parametro di forma lambda della parametrizzazione NSS.
        lambda1: numero decimale 1 x 1, rappresentanta il secondo parametro di forma lambda della parametrizzazione NSS.
        TimeVec: ndarray n x 1 di scadenze per cui sono stati osservati i rendimenti in YieldVec.
        YieldVec: ndarray n x 1 di rendimenti osservati.
        
    Restituisce:
        array 6 x 1 di parametri e fattori che si adattano meglio ai rendimenti osservati.
        
    Fonti:
    - https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
    - https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    
    Implementato da Gregor Fabjan di Qnity Consultants il 11/07/2023
    """
 
    bounds = [(None, None)] * 4 + [(1e-6, None)] * 2 # i parametri di forma lambda0 e lambda1 devono essere positivi
    opt_sol = minimize(NSSGoodFit, x0=np.array([beta0, beta1, beta2, beta3, lambda0, lambda1]), args = (TimeVec, YieldVec), method="Nelder-Mead", bounds = bounds)
    if (opt_sol.success):
        return opt_sol.x
    else:
        raise RuntimeError("L'ottimizzazione Nelder-Mead non è convergente: " + str(opt_sol.message))
