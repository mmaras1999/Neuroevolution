import numpy as np
import scipy
import scipy.linalg
import plotly.graph_objs as go
import plotly.express as px

class CMA_ES_Active:
        
    def __init__(self, x0, sigma, maxfevals = 10000, popsize = None, weights = None):
        N = x0.shape[0]
        self.N = N
        
        self.chiN = N**0.5 * (1 - 1. / (4 * N) + 1. / (21 * N**2))
        
        self.lam = 4 + int(3 * np.log(N)) if not popsize else popsize
        print(f"Popsize: {self.lam}")
        
        self.µ = int(self.lam / 2)
        
        if weights:
            self.weights = weights
        else:
            self.weights = np.array([np.log(self.lam / 2 + 0.5) - np.log(i + 1) for i in range(self.lam)])

        self.µeff = np.sum(self.weights[:self.µ])**2 / np.sum(self.weights[:self.µ]**2)
        µeffneg = np.sum(self.weights[self.µ:]) ** 2 / np.sum(self.weights[self.µ:]**2)
        
        self.cc = (4 + self.µeff/N) / (N+4 + 2 * self.µeff/N)
        self.cs = (self.µeff + 2) / (N + self.µeff + 5)
        self.c1 = 2 / ((N + 1.3)**2 + self.µeff) 
        self.cµ = min([1 - self.c1, 2 * (self.µeff - 2 + 1/self.µeff) / ((N + 2)**2 + self.µeff)])
        self.damps = 2 * self.µeff/self.lam + 0.3 + self.cs
        
        self.weights[:self.µ] /= np.sum(self.weights[:self.µ])
        
        aµ = 1 + self.c1 / self.cµ
        aµeff = 1 + 2 * µeffneg / (self.µeff + 2)
        aposdef = (1 - self.c1 - self.cµ) / self.N / self.cµ 
        
        self.weights[self.µ:] /= np.sum(-self.weights[self.µ:]) / min(aµ, aµeff, aposdef)
        
        self.xmean = x0[:]
        self.sigma = sigma
        self.C = np.identity(N)
        
        self.pc = np.zeros(N) 
        self.ps = np.zeros(N) 
        self.lazy_gap_evals = 0.5 * N * self.lam * (self.c1 + self.cµ)**-1 / N**2
        self.maxfevals = maxfevals
      
        self.counteval = 0 
        self.fitvals = []   
        self.best = (x0, None)
        self.condition_number = 1
        self.eigen_values = np.ones(N)
        self.eigen_vectors = np.identity(N)
        self.updated_eval = 0
        self.inv_sqrt = np.identity(N)

    def _update_eigensystem(self, current_eval, lazy_gap_evals):
        if current_eval <= self.updated_eval + lazy_gap_evals:
            return self
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.C)
        self.inv_sqrt = self.eigen_vectors @ np.diag(self.eigen_values**-0.5) @ self.eigen_vectors.T
        self.condition_number = self.eigen_values.max() / self.eigen_values.min()
        self.updated_eval = current_eval
         
    def sample(self):
        """Wylosuj probke nowych osobnikow"""
        return self.xmean + self.sigma * np.dot(np.random.randn(self.lam, self.N), np.linalg.cholesky(self.C).T)
    
    def update(self, x, fitvals):
        """Zaktualizuj wartosci uzyskanych parametrow"""
        self.counteval += fitvals.shape[0] # Zwiekszamy licznik wykonan
        N = self.N
        x_old = self.xmean.copy()
        
        # Posortuj osobniki po wartosci funkcji celu
        ids = np.argsort(fitvals)
        x = x[ids]
        self.fitvals = fitvals[ids]
        
        self.best = (x[0], self.fitvals[0])
        
        self.xmean = (self.weights[:self.µ] @ x[:self.µ]).ravel()
        y = (x - x_old) / self.sigma
        
        # Aktualizacja sciezki ewolucji dla sigmy
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.µeff) * self.inv_sqrt @ ((self.xmean - x_old) / self.sigma)
 
        # Aktualizacja sciezki ewolucji dla macierzy kowariancji
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.µeff) * ((self.xmean - x_old) / self.sigma)
        
        #Aktualizacja macierzy kowariancji
        self.C = (1 - self.c1 - self.cµ * self.weights.sum()) * self.C
        self.C += self.c1 * np.outer(self.pc, self.pc) #+ self.cµ * (y.T @ np.diag(self.weights) @ y)
        
        for r in range(self.lam):
            self.C += np.outer(y[r], y[r]) * self.cµ * self.weights[r]
        
        self.C = (self.C + self.C.T) / 2.0 # Upewniamy sie, ze macierz jest symetryczna
        
        # Aktualizacja rozmiaru kroku
        
        self.sigma = self.sigma * np.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        self._update_eigensystem(self.counteval, self.lazy_gap_evals)
        
    def terminate(self):
        """Zakoncz algorytm"""
        if self.counteval <= 0:
            return False
        if self.counteval >= self.maxfevals:
            return True
        if self.condition_number > 1e10:
            return True
        if self.sigma * np.max(self.eigen_values)**0.5 < 1e-10:
            return True
        return False
    
def optimize_active(func, x0, sigma, maxfevals = 1000000, popsize = None, weights = None):
    cma_es = CMA_ES_Active(x0, sigma, maxfevals, popsize, weights)
    res = []
    cntr = 0
    while not cma_es.terminate():
        cntr+=1
        x = cma_es.sample()
        f_eval = func(x)
        cma_es.update(x, f_eval)
        res.append(cma_es.best)
        if cntr % 100 == 0:
            print(f"Iteration {cntr:5d}: {res[-1][1]}")
    return res

def optimize_and_plot_active(f, sigma = 1, d = 10, popsize = None):
    x0 = np.repeat(10.0, d)
    res = optimize_active(f, x0, sigma, popsize = popsize)
    print(f"Best: {res[-1][0]}, value: {res[-1][1]}")
    y = np.array([nd for st, nd in res])
    fig = px.line(x = np.arange(y.shape[0]) + 1, y = y)
    fig.show()