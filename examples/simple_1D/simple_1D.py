import torch
import numpy as np
from matplotlib import pyplot as plt
from wskde.wskde import WSKDE

if __name__ == "__main__":
    # Define the ground truth success rate function, and sample for use in visualization
    lower = -10 #lower bound on parameter space
    upper = 10 #upper bound on the parameter space
    n_samples_gt = 100 #number of samples used in the visualization
    x_gt = torch.Tensor(torch.linspace(lower,upper,n_samples_gt))
    y_gt = torch.sigmoid(x_gt)

    # Generate a random dataset by:
    # 1) randomly sampling parameter values
    # 2) computing the success rate for each value
    # 3) Performing Bernoulli trials using the computed success rates
    n_samples_rand = 50
    x_rand = lower + (upper-lower)*torch.rand((n_samples_rand))
    prob_rand = torch.sigmoid(x_rand)
    y_rand = torch.bernoulli(prob_rand)
    x_true = x_rand[torch.nonzero(y_rand)].squeeze()
    x_false = x_rand[torch.nonzero(y_rand == 0)].squeeze()

    # Compute WilsonScore KDE bounds for the given parameter values
    # 1) Define the desired bandwidth matrix for the Gaussian kernel
    #    - In this 1D example we use a 1x1 bandwidth kernel 
    #    - Hint: It is suggested to use a bandwidth kernel that is proportional to the 
    #      square root of the covarinace of the data
    # 2) Set the training data
    #    - Note: WSKDE doesn't have a training phase, but uses a supervised data set
    #      to perform online inference from all points
    h = 0.5
    H = torch.diag(torch.Tensor([h]))
    wskde = WSKDE(H)
    wskde.set_training_samples(x_rand,y_rand)
    x_wskde = x_gt
    p_wskde, sigma_wskde = wskde(x_wskde,z=1.96)


    # Plot the ground truth success rate function, the random samples and the WSKDE bounds
    plot_displacement = -0.05
    fig, ax = plt.subplots() 
    ax.plot(x_gt.cpu().numpy(),y_gt.cpu().numpy(),'--k')
    ax.plot(x_true.cpu().numpy(), plot_displacement*np.ones(x_true.shape),'.g')
    ax.plot(x_false.cpu().numpy(), plot_displacement*np.ones(x_false.shape),'xr')
    ax.plot(x_wskde,p_wskde)
    ax.fill_between(x_wskde,p_wskde-sigma_wskde,p_wskde+sigma_wskde,alpha=0.2)

    plt.show()
