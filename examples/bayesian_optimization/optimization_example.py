import torch
from wskde.wskde import WSKDE
import util
import bayesian_optimization

f = util.IROS25_function(b=0.03)
max_iter = 3000
d_steps=315 #Discretization of the domain used in the grid search

sampler = util.BinomialSampler(f)
# sampler = util.BetaSampler(f,ab=5) #uncomment to use the Beta sampler

bo = bayesian_optimization.BayesianOptimizer(d_steps=d_steps,max_iter=max_iter,h=0.02, kde_method="ws_kde",visualize_each_iteration=False)
result = bo.optimize(sampler)


#Extract results of the optimization
x = result["x"]
y = result["y"]
h = result['h']
z = result['z']
x_lim = result["x_lim"]
x_i_viz = result["x_i"]
p = result["p"]
x_best = x_i_viz[torch.argmax(p,dim=1)][-1]

print("The optimal x is after " + str(max_iter) + " optimizations is:")
print("x_best: " + str(x_best))

#visualize the function estimate at the end of the optimization
d_steps_gt = 1000 #discretization used in the visualization
x_i_viz = torch.linspace(x_lim[0],x_lim[1],d_steps_gt)
wskde = WSKDE(torch.diag(torch.Tensor([h])))
wskde.set_training_samples(x,y)
m, sigma = wskde(x_i_viz,"ws_kde",z)
y_gt = sampler.expectation(x_i_viz)
util.visualize_wskde(x_i_viz,m,m+sigma,m-sigma,x, y, x_i_viz,y_gt,figsize=(24,6),save_to=None)