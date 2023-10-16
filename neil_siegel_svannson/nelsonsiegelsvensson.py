import numpy as np
from scipy.optimize import minimize

def NelsonSiegelSvansson(T, beta0, beta1, beta2, beta3, lambda0, lambda1):
    """
    NelsonSiegelSvansson calcola la curva interpolata/estrappolata nei punti dell'array "T" utilizzando l'algoritmo di Nelson-Siegel-Svannson (NSS),
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
        
    Implementato da Gregor Fabjan di Qnity Consultants il 16/11/2023
    """
    alpha1 = (1 - np.exp(-T / lambda0)) / (T / lambda0)
    alpha2 = alpha1 - np.exp(-T / lambda0)
    alpha3 = (1 - np.exp(-T / lambda1)) / (T / lambda1) - np.exp(-T / lambda1)

    return beta0 + beta1 * alpha1 + beta2 * alpha2 + beta3 * alpha3


def NSSGoodFit(params, TimeVec, YieldVec):
    """
    NSSGoodFit calcola i residui tra il rendimenti osservati nel mercato e quelli previsti dall'algoritmo NSS con la parametrizzazione specificata.
    
    Argomenti:
        params: tuple 6 x 1 continene i 6 parametri dell'algoritmo NSS. La sequenza dei parametri deve essere (beta0, ..., beta4, lambda0, lambda1).
        TimeVec: ndarray n x 1 di scadenze per cui sono stati osservati i rendimenti in YieldVec.
        YieldVec: ndarray n x 1 di rendimenti osservati.
        
    Restituisce:
        float 1 x 1, distanza euclidea tra i punti calcolati e i dati osservati.
        
    Implementato da Gregor Fabjan di Qnity Consultants il  16/11/2023
    """

    return np.sum((NelsonSiegelSvansson(TimeVec, params[0], params[1], params[2], params[3], params[4], params[5])-YieldVec)**2)

def NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, TimeVec, YieldVec):
    """
    NSSMinimize utilizza la funzione di minimizzazione incorporata nella libreria scipy di Python. La funzione configura i parametri e la funzione NSSGoodFit in modo
    che sia compatibile con il modo in cui la funzione minimize richiede i suoi argomenti. Se l'ottimizzazione non converge, l'output è un array vuoto.
    
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
        array 6 x 1 di parametri e fattori che si adattano meglio ai rendimenti osservati (o un array vuoto se l'ottimizzazione non è riuscita).
        
    Fonti:
    - https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
    - https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    
    Implementato da Gregor Fabjan di Qnity Consultants il 11/07/2023
    """
 
    opt_sol = minimize(NSSGoodFit, x0=np.array([beta0, beta1, beta2, beta3, lambda0, lambda1]), args = (TimeVec, YieldVec), method="Nelder-Mead")
    if (opt_sol.success):
        return opt_sol.x
    else:
        return []
