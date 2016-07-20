print(__doc__)

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
# Licence: BSD 3 clause
from mpl_toolkits.mplot3d import *
import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
from matplotlib import cm

# Standard normal distribution functions
phi = stats.distributions.norm().pdf
PHI = stats.distributions.norm().cdf
PHIinv = stats.distributions.norm().ppf

# A few constants
lim = 8

class GaussianProcessModel(object):
    def __init__(self,lim=1,res=100):
        self.res=100
        self.lim=lim
        x1, x2 = np.meshgrid(np.linspace(- 0*self.lim, self.lim, self.res),
                     np.linspace(- 0*self.lim, self.lim, self.res))
        self.xgrid = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T
        
    def simulatedsurface(self,x):
        """The function to predict"""
        return 5 - x[:, 1] - .5* x[:, 0] ** 2.

    def averageduplicates(self, allpos,allmeas):
        #get rid of duplicate points
        uniquepos = np.array(list(set(tuple(p) for p in allpos)))
        #average measurements at duplicated points
        uniquemeas = np.empty(uniquepos.shape[0])
        for i, pos in enumerate(uniquepos):
            uniquemeas[i] = np.mean(allmeas[(allpos == pos).all(1)])

        return [uniquepos, uniquemeas]   

    def update_GP(self,meas_locations):
        # define grid to evaluate stuff on
        #res = 100
        sensornoise=.1
        noise= sensornoise*np.random.randn(meas_locations.shape[0])
        measurements = self.simulatedsurface(meas_locations) + noise
        print meas_locations
        print measurements
        [self.meas_locations,self.new_measurements]=self.averageduplicates(meas_locations,measurements)
        # Instanciate and fit Gaussian Process Model
        gp = GaussianProcess(corr='squared_exponential',
            #theta0=10e-1,
            thetaL=1e-10, thetaU=1e1,
            nugget=(sensornoise/self.new_measurements) ** 2
        )
        # Observations
        
        print noise
        
        # Don't perform MLE or you'll get a perfect prediction for this simple example!
        gp.fit(self.meas_locations, self.new_measurements)
        #evaluate the prediction and its MSE on a grid
        y_pred, MSE = gp.predict(self.xgrid, eval_MSE=True)
        sigma = np.sqrt(MSE)
        y_pred = y_pred.reshape((self.res, self.res))
        sigma = sigma.reshape((self.res, self.res))
        self.model=y_pred
        self.uncertainty=sigma
        return [y_pred,sigma]

    def plotGP(self):
        X=self.meas_locations
        y_pred=self.model
        sigma=self.uncertainty
        y_true = self.simulatedsurface(self.xgrid).reshape((
            self.res, self.res))
        x1, x2 = np.meshgrid(np.linspace(0, self.lim, self.res),
                             np.linspace(0, self.lim, self.res))
        # Plot the probabilistic classification iso-values using the Gaussian property
        # of the prediction
        # Plot the diagrams
        fig = pl.figure(figsize=(5, 5))
        y = self.simulatedsurface(X)
        # first: plot a selection of unconstrained functions
        ax = fig.add_subplot(221)
        #ax.plot(x, draws.T, '-k')
        #ax.set_ylabel('$f(x)$')
    
        #fig = ax.figure(1)
        #ax = # FIXME: g.add_subplot(111)
        ax.axes.set_aspect('equal')
        pl.xticks([])
        pl.yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pl.xlabel('$x_1$')
        pl.ylabel('$x_2$')

        cax = pl.imshow(y_pred, cmap=cm.gray_r, alpha=0.8,
                        extent=(0, lim, 0, lim))
        norm = pl.matplotlib.colors.Normalize(vmin=0., vmax=0.9)
        cb = pl.colorbar(cax, ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], norm=norm)
        cb.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) \leq 0\\right]$')

        ax = fig.add_subplot(222)
        ax.axes.set_aspect('equal')
        pl.xticks([])
        pl.yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pl.xlabel('$x_1$')
        pl.ylabel('$x_2$')
        cax = pl.imshow(sigma, cmap=cm.gray_r, alpha=0.8,
                        extent=(0, lim, 0, lim))
        norm = pl.matplotlib.colors.Normalize(vmin=0., vmax=0.9)
        cb = pl.colorbar(cax, ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], norm=norm)
        cb.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) \leq 0\\right]$')

        ax = fig.add_subplot(223, projection='3d')
        ax.axes.set_aspect('equal')
        pl.xticks([])
        pl.yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pl.xlabel('$x_1$')
        pl.ylabel('$x_2$')
        ax.plot_surface(x1, x2, y_true, cmap=cm.hot, alpha=.5)
        ax.scatter(X[:,0], X[:,1], y); 
        norm = pl.matplotlib.colors.Normalize(vmin=0., vmax=0.9)

        ax = fig.add_subplot(224, projection='3d')
        ax.axes.set_aspect('equal')
        pl.xticks([])
        pl.yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pl.xlabel('$x_1$')
        pl.ylabel('$x_2$')
        ax.plot_surface(x1, x2, y_pred, color='red', alpha=1)
        ax.scatter(X[:,0], X[:,1], y);
        ax.plot_surface(x1, x2, y_pred+sigma, alpha=.1)
        ax.plot_surface(x1, x2, y_pred-sigma, alpha=.1)

        norm = pl.matplotlib.colors.Normalize(vmin=0., vmax=0.9) 
        pl.show()

# Take meaksurements at these points
X = np.random.uniform(0,1,size=(10,2))


GP=GaussianProcessModel()
#update the GP
GP.update_GP(X)

# Evaluate real function  on a grid
GP.plotGP()
