<h1 align="center" style="border-botom: none">
  <b>
    🐍 Algoritmo Nelson-Siegel-Svannson 🐍     
  </b>
</h1>

</br>

Algoritmo popolare per adattare una curva dei rendimenti a dati osservati.

## Problema
I dati sui rendimenti dei titoli di stato sono normalmente disponibili solo per paio di scadenze, mentre l'utente è di solito interessato in più rendimenti. 
  
## Soluzione
Una soluzione popolare è di utilizzare un algoritmo per trovare una funzione che si adatta ai punti dati esistenti. In questo modo, la funzione può essere utilizzata per interpolare/estrappolare qualsiasi altro punto. Il modello Nelson-Siegel-Svensson è un algoritmo di adattamento di curve abbastanza flessibile da approssimare la maggior parte delle applicazioni reali.

Il Nelson-Siegel-Svensson estende il metodo Nelson-Siegel a 4 parametri a 6 parametri. Svensson ha introdotto due parametri aggiuntivi per adattarsi meglio alla varietà di forme dei tassi forward istantanei o delle curve dei rendimenti osservate nella pratica.

Vantaggi:
-  Produce una curva dei tassi forward scorrevole e ben comportata.
-  Interpretazione intuitivadei parametri. `beta0` è il tasso di interesse a lungo termine e `beta0+beta1` è il tasso a breve termine istantaneo.

Per trovare il valore ottimale dei parametri, viene utilizzato l'algoritmo del simplesso Nelder-Mead (già implementato nel pacchetto scipy). Il link all'algoritmo di ottimizzazione è  Gao, F. and Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters. 2012. Computational Optimization and Applications. 51:1, pp. 259-277.

La formula per la curva dei rendimenti (valore del rendimento per una scadenza al tempo 't') è data dalla formula:


![formula](https://render.githubusercontent.com/render/math?math=\r(t)=\beta_{1}) +
![formula](https://render.githubusercontent.com/render/math?math=\beta_{2})
![formula](https://render.githubusercontent.com/render/math?math=\big(\frac{1-exp(\frac{-t}{\lambda_1})}{\frac{t}{\lambda_1}}\big)) +
![formula](https://render.githubusercontent.com/render/math?math=\beta_{3})
![formula](https://render.githubusercontent.com/render/math?math=\big(\frac{1-exp(\frac{-t}{\lambda_1})}{\frac{t}{\lambda_1}}-exp(\frac{-t}{\lambda_1})\big)) +
![formula](https://render.githubusercontent.com/render/math?math=\beta_{4})
![formula](https://render.githubusercontent.com/render/math?math=\big(\frac{1-exp(\frac{-t}{\lambda_2})}{\frac{t}{\lambda_2}}-exp(\frac{-t}{\lambda_2})\big))

### Parametri

   - Tassi di rendimento osservati `YieldVec`.
   - Scadenza di ciascun rendimento osservato `TimeVec`.
   - Stima iniziale dei parametri `beta0`, `beta1`, `beta2`, `beta3`, `labda0`, and `lambda1`.
   - Scadenze target `TimeResultVec`.

### Output desiderato

   - Tassi di rendimento calcolati per le scadenze di interesse `TimeResultVec`.

## Come iniziare

L'utente è interessato al rendimento previsto per i titoli di stato con scadenze di 1, 2, 5, 10, 25, 30 e 31 anni. Ha dati sui titoli di stato con scadenza di 1, 2, 5, 10 e 25 anni. Il rendimento calcolato per quei titoli è del 0,39%, 0,61%, 1,66%, 2,58% e 3,32%.

  ```bash
from nelsonsiegelsvensson import *
import numpy as np

TimeVec = np.array([1, 2, 5, 10, 25])
YieldVec = np.array([0.0039, 0.0061, 0.0166, 0.0258, 0.0332])
beta0   = 0.1 # stima iniziale
beta1   = 0.1 # stima iniziale
beta2   = 0.1 # stima iniziale
beta3   = 0.1 # stima iniziale
lambda0 = 1 # stima iniziale
lambda1 = 1 # stima iniziale

TimeResultVec = np.array([1, 2, 5, 10, 25, 30, 31]) # Scadenze per i rendimenti di interesse


## Implementazione
OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, TimeVec, YieldVec) # Viene utilizzato l'algoritmo del simplesso Nelder-Mead per trovare i parametri che producono una curva con i minimi residui rispetto ai dati di mercato.

# Stampare la curva dei rendimenti con i parametri ottimali per confrontarla con i dati forniti
print(NelsonSiegelSvansson(TimeResultVec, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3], OptiParam[4], OptiParam[5]))
```
