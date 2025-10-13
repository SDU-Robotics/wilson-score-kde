import torch
import util
from wskde.wskde import WSKDE
from matplotlib import pyplot as plt
from tqdm import tqdm

class AquisitionFunction:
    ''' Defines the interface for aquisition functions '''
    def find_next_idx(self,mu:torch.Tensor,sigma:torch.Tensor)->int:
        '''
        Function for choosing the index of the next sample from the input list of possible samples.
        For each sample the upper and lower confidence bound is given as (mu-sigma) and (m+sigma) 
            mu: shape n
            sigma: shape n
            return int
        '''
        pass

class PruneAndSample(AquisitionFunction):
    ''' This aquisition function first prunes the samples where the upper confidence bound (mu_i+sigma_i)
    is lower than the maximum lower confidence bound S_t = (mu-sigma)_max. It then samples the next index
    by sampling among the non-prunes indexes either using a uniform distribution or weighted sampling using
    the weight w_i = mu + sigma - S_t.
    
    The choice between weighted and uniform distribution is random with a probability set in the constructor
    '''
    def __init__(self, uniform_dist_prob:float = 1):
        ''' uniform_dist_prob: The probability of using a uniform sampling instead of a weighted. The default
        is 1, e.i. always using a uniform sampling 
        '''
        self.uniform_dist_prob = uniform_dist_prob

    def find_next_idx(self,mu:torch.Tensor,sigma:torch.Tensor)->int:
        ''' p: shape n
            sigma: shape n
            return int
            '''
        # Compute threshold from highest lower bound
        lcb = mu-sigma
        lcb[lcb<0] = 0 #enforce confidence bounds within [0,1] e.g. for na_kde
        lcb[lcb>1] = 1
        S_t = torch.max(lcb).item()

        # Set un-normalized probability according to distance from S_t to upper bound
        p = mu + sigma - S_t
        
        # Prune p values with ucb lower than S_t
        p[p<0] = 0

        # Enforce uniform sampling when the p is ill-defined (e.g. in reguar na_kde)
        if p.norm()==0:
            p[:] = 1
        

        # If sampling using a uniform distribution over the non-pruned
        if torch.rand(1).item()<self.uniform_dist_prob:
            p[p>0] = 1

        # Normalize distribution
        p = p/p.norm()

        # sample a single x_i with probabilities given by p
        next_idx = p.multinomial(num_samples=1, replacement=True).item()
        return next_idx

class BayesianOptimizer:
    ''' This is a class for performing WS-KDE based Bayesian Optimization of an arbitrary 1D expectation value function with a stocastic output in the range [0,1]'''
    def __init__(self, max_iter = 100, h = 0.1, d_steps = 101, x_lim=[0,2*torch.pi], z=1.96, kde_method="ws_kde", visualize_each_iteration = False):
        '''
        max_iter: number of iterations in the optimization
        h: length scale (standard deviation) of an isotropic Gaussian kernel used in WS-KDE
        d_steps: number of steps used in the aquisition functions grid search
        x_lim: upper and lower bounds of the search space
        z: the confidence defined as the z-value (the default z=1.96 correspond to 95% confidence bounds)
        kde_method: choice between using WS-KDE ("ws_kde") and regular KDE ("na_kde")
        visualize_each_iteration: Choose whether to visualize the upper and lower bounds of the function estimate in each iteration.
        '''
        self.max_iter = max_iter
        self.x = None
        self.y = None
        self.h = h
        self.d_steps = d_steps #Number of discretization steps
        self.x_lim = x_lim
        self.z = z
        self.kde = WSKDE(torch.diag(torch.Tensor([self.h])))
        self.kde_method=kde_method
        self.aquisition_function = PruneAndSample(1)
        self.visualize_each_iteration = visualize_each_iteration

        #containers for saving data for all steps of the optimization
        self.p_all = None
        self.sigma_all = None


    def find_next_x(self):
        ''' Finds the next x-value to evaluate based on the already evaluated x and y values
        '''
        x_i = torch.linspace(self.x_lim[0], self.x_lim[1], self.d_steps)
        x_i = x_i.reshape((-1,1)) #Add batch x-dimension of one (shape {B,d})
        p_est, sigma_est = self.kde(x_i,self.kde_method,self.z)
        self.p_all = torch.cat((self.p_all,p_est.reshape((1,-1))),dim=0)
        self.sigma_all = torch.cat((self.sigma_all,sigma_est.reshape((1,-1))),dim=0)
        i_next = self.aquisition_function.find_next_idx(p_est,sigma_est)
        x_next = x_i[i_next,:]
        return x_next

    def one_iter(self, sampler:util.Sampler):
        ''' Performs one optimization iteration '''
        #determine x_i+1
        x_next = self.find_next_x()

        #evaluate in x_i+1
        y_next = sampler.sample(x_next.reshape((1,-1)))[0]
        self.x = torch.cat((self.x,x_next.unsqueeze(0)),dim=0)
        self.y = torch.cat((self.y,y_next.unsqueeze(0)),dim=0)

        #update function estimate
        self.kde.set_training_samples(self.x,self.y)
        


    def optimize(self, sampler:util.Sampler):
        ''' Optimize the stocastic function 'sampler'. For more information on how the sampler is defined, see util.Sampler'''
        x_i = torch.linspace(self.x_lim[0], self.x_lim[1], self.d_steps)
        x_i = x_i.reshape((-1,1)) #Add batch dimension

        #First iteration initializes relevant tensors
        x_0 = x_i[torch.randint(self.d_steps,(1,)).item(),:]
        y_0 = sampler.sample(x_0)[0]
        self.x= torch.Tensor([x_0]).reshape((1,1))
        self.y= torch.Tensor([y_0]).reshape((1,1))
        self.kde.set_training_samples(self.x,self.y)
        p,sigma = self.kde(x_i,self.kde_method,self.z)
        self.p_all = p.reshape(1,-1)
        self.sigma_all = sigma.reshape(1,-1)

        for i in tqdm(range(1,self.max_iter)):
            self.one_iter(sampler)
            if self.visualize_each_iteration:              
                p = self.p_all[-1,:]
                sigma = self.p_all[-1,:]
                util.visualize_wskde(x_i.squeeze(),p,p+sigma,p-sigma,self.x,self.y,x_i.squeeze(),sampler.expectation(x_i))

        x_gt = x_i
        y_gt = sampler.expectation(x_i)
        result = {"max_iter":self.max_iter,
                       "x":self.x,
                       "y":self.y,
                       "h":self.h,
                       "d_steps":self.d_steps,
                       "x_lim": self.x_lim,
                       "z":self.z,
                       "x_i": x_i.squeeze(),
                       "p":self.p_all,
                       "sigma":self.sigma_all,
                       "x_gt":x_gt,
                       "y_gt":y_gt}
        return result