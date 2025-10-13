import torch
import numpy as np
from matplotlib import pyplot as plt
import sys

class Func:
    ''' 
    x: torch.Tensor where the last dimension is the d-dimensional point
       e.g. {M x N x d } would be an M x N grid of d-dimensional points
    '''
    def __call__(self, x:torch.Tensor):
        pass

class Cosine(Func):
    """ 1D function """
    def __init__(self, wavelength:float):
        self.wavelength = wavelength

    def __call__(self,x:torch.Tensor):
        return torch.cos(2*torch.pi/self.wavelength*x)

class IROS25_function(Func):
    """ 1D function """
    def __init__(self,a=0.6,b=0.01):
        self.a = a
        self.b = b
    def __call__(self, x):
        return 0.5 *(torch.sin(self.a * torch.pow(x,2.0)) * torch.exp(-self.b *(2*torch.pi - x)) + 1)

class Sampler:
    ''' Interface for the Sampler class. A sampler is a stocastic function f(x)~PDF, where PDF
    is a distribution on the interval [0,1]. A Sampler can either return the expectation value
    at $x$, or sample the stocastic function at $x$'''
    
    def expectation(self,x):
        ''' 
        x: expected to be a Tensor where the last dimension is the d-dimensional point
        e.g. {M x N x d } would be an M x N grid of d-dimensional points
        '''
        pass

    
    def sample(self,x):
        '''
        x: expected to be a Tensor where the last dimension is the d-dimensional point
        e.g. {M x N x d } would be an M x N grid of d-dimensional points
        '''
        pass


class BinomialSampler(Sampler):
    ''' An implementation of Sampler, where the expectation value is a function of $x$ and is defined
    by the Func given in the contructor. The random samples are Bernoulli trials.'''
    def __init__(self,f:Func):
        self.f = f
        
    
    def expectation(self, x):
        return self.f(x)
    
    def sample(self,x):
        return torch.bernoulli(self.f(x))
    
class BetaSampler(Sampler):
    ''' An implementation of Sampler, where the expectation value is a function of $x$ and is defined
    by the Func given in the contructor. The random samples are drawn from a beta distribution where
    a+b is defined in the constructor.
    
    Note: The theoretical expectation of a beta function is E[x] = a/(a+b). The a and b parameters are
     thus uniquely defined by the expectation value and the sum a+b: a=E[x]*(a+b) and b=(a+b)-a)'''
    def __init__(self,f:Func, ab:float):
        self.f = f
        assert(ab>0)
        self.ab = torch.FloatTensor([ab])
    
    def expectation(self, x):
        return self.f(x)
    

    def numerically_stable_a_b(self, expectation, ab, eps=1e-10):
        if expectation == 0:
           expectation = eps
        elif expectation == 1:
            expectation = 1- eps
        a = expectation*ab
        b = ab - a
        if b==0:
            b = eps
        if a==0:
            a = eps
        return a, b


    def sample(self,x):
        ''' 
        x: shape B
        return: shape B

        theoretic expectation is E[x] = a/(a+b). So we can define the distribution by defining both expectation and
        the value of (a+b) => a=E[x]*(a+b) and b=(a+b)-a
        '''
        output = torch.zeros(x.shape)
        for i,xi in enumerate(x):
            exp = self.expectation(xi)
            a,b = self.numerically_stable_a_b(expectation=exp, ab=self.ab)
            dist = torch.distributions.Beta(a,b)
            output[i] = dist.sample()
        return output
    
    def distribution_xy(self, expectation=None, x=None, n_samples=100):
        ''' 
        This method is used to sample xy-values from the beta distribution. The beta distribution to 
        sample from can either be the one defined at x (from the expection function and a+b given in the contructor)
        or be a beta using a manually provided expectiation value (but still using the a+b given in the contructor).

        expectation: Define the expectation value of the beta function

        x: Define the x value. This x-value combined with the function defined in the sampler
           will be used to compute the expectation value.
        n_samples: The number of samples in the range [0,1] for which the beta
                   distribution should be computed
        return: vector of x,y values used to plot the beta function

        theoretic expectation is E[x] = a/(a+b). So we can define the distribution by defining both expectation and
        the value of a => b = a/E[x] - a '''

        if (expectation is None and x is None) or (expectation is not None and x is not None):
            raise ValueError("exactly one of expection and x must be set")
        if expectation is not None:              
            exp = expectation
        if x is not None:
            exp = self.expectation(x)

        a,b = self.numerically_stable_a_b(exp,self.ab)
        dist = torch.distributions.Beta(a,b)
        
        x = torch.linspace(0,1,n_samples)
        return x, dist.log_prob(x).exp()
    
def visualize_wskde(x_i,m,upper,lower,x_data,y_data,x_gt,y_gt,figsize=None, save_to=None):
    ''' Method for visualizing the estimated confidence bounds vs. the ground truth S(x) '''
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.fill_between(x_i,lower,upper,alpha=0.2, label="Conf. bounds")
    ax.scatter(x_data, y_data, color='r',marker=".", label="Data points", alpha=0.1)
    ax.plot(x_i,m,color='k')
    ax.plot(x_gt, y_gt, color='b', label="S(x)")
    ax.legend(fontsize=26, loc='lower left')
    ax.tick_params(labelsize=26)
    plt.ylim([0,1])
    if save_to is not None:
        fig.savefig(save_to)
    plt.show()