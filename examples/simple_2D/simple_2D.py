import torch
import numpy as np
from matplotlib import pyplot as plt
from wskde.wskde import WSKDE

if __name__ == "__main__":
    # Define the ground truth success rate function, and sample for use in visualization
    def sigmoid2d(x1, x2):
        return torch.sigmoid(-15 + 5*torch.sqrt(x1 * x1 + x2 * x2))
    
    visualize = True
    n_1d = 30
    lower = -5
    upper = 5
    x1s = torch.linspace(lower, upper, steps=n_1d)
    x2s = torch.linspace(lower, upper, steps=n_1d)
    x1_gt, x2_gt = torch.meshgrid(x1s, x2s, indexing='xy')
    y_gt = sigmoid2d(x1_gt,x2_gt)
    n_samples_gt = n_1d*n_1d
    if visualize:
        ax = plt.axes(projection='3d')
        ax.plot_surface(x1_gt.numpy(), x2_gt.numpy(), y_gt.numpy())

    # Generate a random dataset by:
    # 1) randomly sampling parameter values
    # 2) computing the success rate for each value
    # 3) Performing Bernoulli trials using the computed success rates
    n_samples_rand = 25000
    x1_rand = lower + (upper-lower)*torch.rand((n_samples_rand))
    x2_rand = lower + (upper-lower)*torch.rand((n_samples_rand))
    x_rand = torch.stack((x1_rand,x2_rand)).T #shape (n,2)
    prob_rand = sigmoid2d(x1_rand,x2_rand)
    y_rand = torch.bernoulli(prob_rand) #shape (n)
    if visualize:
        ax.scatter(x_rand[:,0], x_rand[:,1], y_rand, color='r')
        plt.show()

    # Compute WilsonScore KDE bounds for the given parameter values
    # 1) Define the desired bandwidth matrix for the Gaussian kernel
    #    - In this 2D example we use a 2x2 isotropic bandwidth kernel
    # 2) Set the training data
    #    - Note: WSKDE doesn't have a training phase, but uses a supervised data set
    #      to perform online inference from all points
    h = 0.1
    wskde = WSKDE(torch.diag(torch.Tensor([h,h])))
    wskde.set_training_samples(x_rand,y_rand)
    x_gt = torch.reshape(torch.stack((x1_gt, x2_gt)),(2,-1)).T
    x_wskde = x_gt
    p_wskde, sigma_wskde = wskde(x_wskde,z=1.96)


    # Plot the ground truth success rate function, and the WSKDE bounds as both
    # Contour plots and scatter plots
    x1 = x_gt[:,0].reshape(n_1d,n_1d) # convert x_gt, p_wskde, and sigma_wskde into the original meshgrid
    x2 = x_gt[:,1].reshape(n_1d,n_1d)
    p_con = p_wskde.reshape(n_1d,n_1d)
    sigma_con = sigma_wskde.reshape(n_1d,n_1d)

    plt.figure()
    plt.contour(x1,x2,p_con, levels=[0.5], colors='blue')
    plt.contour(x1,x2,p_con -sigma_con, levels=[0.5], colors='red')
    plt.contour(x1,x2,p_con +sigma_con, levels=[0.5], colors='green')
    plt.contour(x1,x2,y_gt, levels=[0.5],colors='black')
    plt.show()
    
    plt.figure()
    ax_estimate = plt.axes(projection='3d')
    ax_estimate.scatter(x_gt[:,0], x_gt[:,1], p_wskde, color='b')
    ax_estimate.scatter(x_gt[:,0], x_gt[:,1], p_wskde+sigma_wskde, color='g')
    ax_estimate.scatter(x_gt[:,0], x_gt[:,1], p_wskde-sigma_wskde, color='r')
    plt.show()

