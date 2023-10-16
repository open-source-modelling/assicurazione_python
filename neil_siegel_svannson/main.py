"""
Il algoritmo Nelson-Siegel-Svensson è una popolare estensione del metodo 
Nelson-Siegel a 4 parametri, che aggiunge 2 parametri ulteriori, portando il totale a 6. 
Un algoritmo per l'interpolazione/estrapolazione della curva dei rendimenti insieme a altre applicazioni. 
Svensson introduce due parametri aggiuntivi per adattarsi meglio alla varietà di forme della curva istantea del tasso forward
o delle curve dei rendimenti osservate nella pratica. 
Una proprietà desiderabile del modello è che produce una curva dei tassi forward scorrevole
 e ben comportata. 
 Un'altra proprietà desiderabile è l'interpretazione intuitiva dei parametri: 
 beta0 è il tasso di interesse a lungo termine e beta0+beta1 è il tasso a breve termine istantaneo. 
 Per trovare il valore ottimale dei parametri, viene utilizzato l'algoritmo del simplesso 
 Nelder-Mead (già implementato nel pacchetto scipy). 
 Il link all'algoritmo di ottimizzazione è il seguente: Gao, F. and Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters.
 2012. Computational Optimization and Applications. 51:1, pp. 259-277
"""

from nelsonsiegelsvensson import *
import numpy as np

## Input
#   - Tassi di rendimento osservati (YieldVec)
#   - Scadenza di ciascun tasso di rendimento osservato (TimeVec)
#   - Stima iniziale dei parametri (beta0, beta1, beta2, beta3, lambda0 e lambda1)
#   - Scadenze target (TimeResultVec)

TimeVec = np.array([1, 2, 5, 10, 25])
YieldVec = np.array([0.0039, 0.0061, 0.0166, 0.0258, 0.0332])
beta0   = 0.1 # stima iniziale
beta1   = 0.1 # stima iniziale
beta2   = 0.1 # stima iniziale
beta3   = 0.1 # stima iniziale
lambda0 = 1 # stima iniziale
lambda1 = 1 # stima iniziale

TimeResultVec = np.array([1, 2, 5, 10, 25, 30, 31])  # Scadenze per i rendimenti di interesse

## Implementazione
OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, TimeVec, YieldVec) # Viene utilizzato l'algoritmo del simplesso Nelder-Mead per trovare i parametri che producono una curva con i minimi residui rispetto ai dati di mercato.

# Stampare la curva dei rendimenti con i parametri ottimali per confrontarla con i dati forniti
print(NelsonSiegelSvansson(TimeResultVec, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3], OptiParam[4], OptiParam[5]))