import numpy as np
import scipy
import scipy.linalg

class LM_MA_ES:
        
    def __init__(self, x0, sigma, maxfevals = 10000, popsize = None):
        N = x0.shape[0]
        self.N = N
        self.chiN = N**0.5 * (1 - 1. / (4 * N) + 1. / (21 * N**2))
        
        self.lam = 4 + int(3 * np.log(N)) if not popsize else popsize
        self.µ = int(self.lam / 2)
        print(f"Popsize: {self.lam}")
        
        self.weights = np.array([np.log(self.µ / 2 + 0.5) - np.log(i + 1) if i < self.µ else 0
                    for i in range(self.lam)])
        self.weights /= np.sum(self.weights)

        self.µeff = 1.0 / np.sum(self.weights**2)
        
        self.m = 4 + int(3 * np.log(N))
        self.cs = 2 * self.lam / N
        self.cc = np.array([self.lam / (np.power(4.0, i) * N) for i in range(self.m)])
        self.cd = np.array([1.0 / (np.power(1.5, i) * N) for i in range(self.m)])
        
        self.xmean = x0[:]
        self.sigma = sigma
        self.mi = [np.zeros(N) for i in range(self.m)]
        self.ps = np.zeros(N)
        self.maxfevals = maxfevals
      
        self.counteval = 0 
        self.fitvals = []   
        self.best = (x0, None)
        self.bestgens = []
        self.genmeans = []
        self.gen = 0
         
    def sample(self):
        self.z = np.random.standard_normal((self.lam, self.N))
        self.d = self.z.copy()
        
        for i in range(self.lam):
            for j in range(min(self.gen, self.m)):
                self.d[i] = (1.0 - self.cd[j]) * self.d[i] + self.cd[j] * self.mi[j] * np.inner(self.mi[j], self.d[i])
        
        return self.xmean + self.sigma * self.d
    
    def update(self, x, fitvals):
        self.counteval += fitvals.shape[0] # update evaluation counter
        N = self.N
        x_old = self.xmean.copy()
        
        # sort individuals by fitness
        ids = np.argsort(fitvals)
        self.fitvals = fitvals[ids]
        x = x[ids]
        self.z = self.z[ids]
        self.d = self.d[ids]

        self.bestgens.append(self.fitvals[0])
        self.genmeans.append(fitvals.mean())
        
        self.best = (x[0], self.fitvals[0])
        self.xmean = (self.weights @ x)
        zmean = (self.weights @ self.z)
        
        # update sigma evolution path
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.µeff) * zmean
        
        for i in range(self.m):
            self.mi[i] = (1.0 - self.cc[i]) * self.mi[i] + np.sqrt(self.µeff * self.cc[i] * (2.0 - self.cc[i])) * zmean

        # sigma update
        self.sigma = self.sigma * np.exp(self.cs / 2.0 * (np.linalg.norm(self.ps) / self.chiN - 1.0))
        self.gen += 1
        