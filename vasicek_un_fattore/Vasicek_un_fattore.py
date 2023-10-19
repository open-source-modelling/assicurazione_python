import numpy as np
import pandas as pd

def simula_vasicek_un_fattore(r0: float = 0.1, a: float = 1.0, lam: float = 0.1, sigma: float = 0.2, T: int = 52, dt = 0.1) -> pd.DataFrame:
    """ Simula una serie temporale di tassi di interesse utilizzando il modello Vasicek a un fattore.
     simulazione_tassi_interesse = simula_vasicek_un_fattore(r0, a, lam, sigma, T, dt)
    
     Argomenti:
       r0 (float): tasso di interesse iniziale del processo di Vasicek
       a (float): parametro di "velocità di reversione" che caratterizza la velocità con cui tali traiettorie si riuniranno intorno if parametro lam col passare del tempo
       lam (float): livello medio a lungo termine attorno al quale si svilupperanno tutte le future traiettorie
       sigma (float): la volatilità istantanea misura istante per istante l'ampiezza dell'aleatorietà che entra nel sistema
       T (int): tempo di modellazione finale. Da 0 a T, la serie temporale viene eseguita
       dt (float): incremento di tempo in cui si svolge il processo. Ad esempio, dt = 0.1 significa che la serie temporale è 0, 0.1, 0.2,...
    
     Restituisce:
       DataFrame Pandas di dimensioni N x 2, in cui l'indice rappresenta il tempo di modellazione e i valori sono una realizzazione del prezzo sottostante
    
     Esempio:
       Modellare il tasso di interesse, che oggi è del 10%. La volatilità istantanea annualizzata è del 20%. L'analisi esterna indica che il parametro di reversione media è 1 e il livello di tasso di interesse a lungo termine è del 10%, quindi la correzione di reversione media è theta = 10% * 1 = 10%. L'utente è interessato a una proiezione dei tassi di interesse per i prossimi 10 anni con incrementi di 6 mesi (0,5 anni).
    
       import pandas as pd
       import numpy as np
    
       simula_vasicek_un_fattore(0.1, 1.0, 0.1, 0.2, 10, 0.5)   
       [output] = Tempo    Tassi di Interesse                
               0.000000        0.100000
               0.526316        0.212055
               1.052632        0.115934
               1.578947        0.012870
               2.105263        0.003295
               2.631579        0.206635
               3.157895        0.191319
               3.684211        0.108299
               4.210526        0.094983
               4.736842        0.075903
               5.263158        0.229143
               5.789474       -0.111977
               6.315789        0.120245
               6.842105        0.116082
               7.368421        0.230879
               7.894737        0.266821
               8.421053        0.229788
               8.947368        0.304378
               9.473684        0.217760
               10.000000       0.217147
     Per ulteriori informazioni, vedere https://en.wikipedia.org/wiki/Vasicek_model
    """
    
    N = int(T / dt) + 1 # numero di punti finali degli sottointervalli di lunghezza 1/dt tra 0 e il tempo di modellazione massimo T

    tempo, delta_t = np.linspace(0, T, num = N, retstep = True)

    r = np.ones(N) * r0

    for t in range(1,N):
        r[t] = r[t-1] * np.exp(-a*dt)+lam*(1-np.exp(-a*dt))+sigma*np.sqrt((1-np.exp(-2*a*dt))/(2*a))* np.random.normal(loc = 0,scale = 1)

    dict = {'Tempo' : tempo, 'Tasso' : r}

    simulazione = pd.DataFrame.from_dict(data = dict)
    simulazione.set_index('Tempo', inplace = True)

    return simulazione
