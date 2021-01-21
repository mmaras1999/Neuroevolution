class MA_ES:
        
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

        self.µeff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        
        self.cc = (4 + self.µeff/N) / (N+4 + 2 * self.µeff/N)
        self.cs = (self.µeff + 2) / (N + self.µeff + 5)
        self.c1 = 2 / ((N + 1.3)**2 + self.µeff) 
        self.cµ = min([1 - self.c1, 2 * (self.µeff - 2 + 1/self.µeff) / ((N + 2)**2 + self.µeff)])
        self.damps = 2 * self.µeff/self.lam + 0.3 + self.cs
        
        self.xmean = x0[:]
        self.sigma = sigma
        self.M = np.identity(N)
        self.ps = np.zeros(N)
        self.maxfevals = maxfevals
      
        self.counteval = 0 
        self.fitvals = []   
        self.best = (x0, None)
        self.bestgens = []
        self.genmeans = []
         
    def sample(self):
        self.z = np.random.randn(self.lam, self.N)
        self.d = np.dot(self.z, self.M.T)
        
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

        self.bestgens.append(fitvals[0])
        self.genmeans.append(fitvals.mean())
        
        self.best = (x[0], self.fitvals[0])
        self.xmean = (self.weights @ x)
        zmean = (self.weights @ self.z)
        
        # update sigma evolution path
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.µeff) * zmean
        
        # M matrix update
        self.M = (1 - self.c1 - self.cµ) * self.M + self.c1 / 2.0 * np.outer(np.dot(self.ps, self.M.T), self.ps)
        for i in range(self.µ):
            self.M += self.cµ / 2.0 * self.weights[i] * np.outer(self.z[i], self.d[i])

        # sigma update
        self.sigma = self.sigma * np.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chiN - 1))
        
    def terminate(self):
        """Zakoncz algorytm"""
        if self.counteval <= 0:
            return False
        if self.counteval >= self.maxfevals:
            return True
        return False