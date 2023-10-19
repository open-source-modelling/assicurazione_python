<h1 align="center" style="border-botom: none">
  <b>
    🐍 Modello Vasicek Un Fattore 🐍     
  </b>
</h1>


Modello di Vasicek con a un fattore di stochasticita per simulare l'evoluzione di redimenti economici. Il modello Vasicek assume che il processo si evolva come un processo Ornstein-Uhlenbeck. Ornstein-Uhlenbeck è un processo stocastico in cui nel tempo il processo tende a derivare verso una media a lungo termine (mean reverting).

## Problema
When trying to simulate for example credit spreads, there is a variety of models. The choice of the model and its limitations are a key factor in deciding which model to implement. There are compelling economic arguments in favour of mean reversion.  

Quando si cerca di simulare, ad esempio, i tassi di spread di credito, esistono diverse tipologie di modelli tra cui scegliere. La scelta del modello è uno dei fattori chiave nella decisione su quale modello schegliere.

## Solution
Uno dei modelli più semplici, il [modello a un fattore Vasicek](https://en.wikipedia.org/wiki/Vasicek_model) assume che il mercato del credito possa essere descritto con un processo stocastico di reversione alla media con una fonte di incertezza derivante da una [moto Browniana](https://en.wikipedia.org/wiki/Brownian_motion). Un proprieta limite di questo modello è che, a causa della distribuzione normale del rumore stochastico, il processo ammette spread negativi, il che potrebbe non essere desiderabile in alcune circostanze. 

L'equazione differenziale stocastica (EDS) del modello Vasicek è mostrata sulla [pagina Wiki](https://en.wikipedia.org/wiki/Vasicek_model) 

### Input

  - `r0` (float): tasso di interesse iniziale del processo Vasicek.
  - `a` (float): parametro di "velocità di reversione" che caratterizza la velocità con cui tali traiettorie si riuniranno intorno a `lam` nel tempo.
  - `lam` (float): livello medio a lungo termine. Tutte le future traiettorie di r si evolveranno intorno a questo livello medio nel lungo periodo.
  - `sigma` (float): la volatilità istantanea misura istante per istante l'ampiezza dell'aleatorietà che entra nel sistema.
  - `T` (integer): tempo di modellazione finale. Da 0 a T, la serie temporale si sviluppa.
  - `dt` (float): incremento di tempo in cui si esegue il processo. Ad esempio, se dt = 0.1, la serie temporale è 0, 0.1, 0.2,...

### Output

 - N x 2 Pandas DataFrame con un percorso di esempio come valori e il tempo di modellazione come indice.

## Getting started
```python
import numpy as np
import pandas as pd
from Vasicek_un_fattore import simula_vasicek_un_fattore

r0 = 0.1    # Tasso di interesse iniziale
a = 1.0     # Parametro di velocità di reversione
lam = 0.1   # Correzione del livello medio del tasso di interesse a lungo termine
sigma = 0.2 # Volatilità istantanea
T = 52      # Tempo di modellazione finale
dt = 0.1    # Incrementi di tempo

print(simula_vasicek_un_fattore(r0, a, lam, sigma, T, dt))
```
