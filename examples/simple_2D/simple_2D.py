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
    x1_test, x2_test = torch.meshgrid(x1s, x2s, indexing='xy')
    y_gt = sigmoid2d(x1_test,x2_test)
    n_samples_gt = n_1d*n_1d
    if visualize:
        ax = plt.axes(projection='3d')
        ax.plot_surface(x1_test.numpy(), x2_test.numpy(), y_gt.numpy())

    # Generate a random dataset by:
    # 1) randomly sampling parameter values
    # 2) computing the success rate for each value
    # 3) Performing Bernoulli trials using the computed success rates
    n_samples_rand = 25000
    x1_train = lower + (upper-lower)*torch.rand((n_samples_rand))
    x2_train = lower + (upper-lower)*torch.rand((n_samples_rand))
    x_train = torch.stack((x1_train,x2_train)).T #shape (n,2)
    prob_rand = sigmoid2d(x1_train,x2_train)
    y_train = torch.bernoulli(prob_rand) #shape (n)
    if visualize:
        ax.scatter(x_train[:,0], x_train[:,1], y_train, color='r')
        plt.show()

    # Compute WilsonScore KDE bounds for the given parameter values
    # 1) Define the desired bandwidth matrix for the Gaussian kernel
    #    - In this 2D example we use a 2x2 isotropic bandwidth kernel
    # 2) Set the training data
    #    - Note: WSKDE doesn't have a training phase, but uses a supervised data set
    #      to perform online inference from all points
    h = 0.1
    wskde = WSKDE(torch.diag(torch.Tensor([h,h])))
    wskde.set_training_samples(x_train,y_train)
    x_test = torch.reshape(torch.stack((x1_test, x2_test)),(2,-1)).T
    p_test, sigma_test = wskde(x_test,z=1.96)


    # Plot the ground truth success rate function, and the WSKDE bounds as both
    # Contour plots and scatter plots
    x1 = x_test[:,0].reshape(n_1d,n_1d) # convert x_gt, p_wskde, and sigma_wskde into the original meshgrid
    x2 = x_test[:,1].reshape(n_1d,n_1d)
    p_contour = p_test.reshape(n_1d,n_1d)
    sigma_contour = sigma_test.reshape(n_1d,n_1d)

    plt.figure()
    plt.contour(x1,x2,p_contour, levels=[0.5], colors='blue')
    plt.contour(x1,x2,p_contour -sigma_contour, levels=[0.5], colors='red')
    plt.contour(x1,x2,p_contour +sigma_contour, levels=[0.5], colors='green')
    plt.contour(x1,x2,y_gt, levels=[0.5],colors='black')
    plt.show()
    
    plt.figure()
    ax_estimate = plt.axes(projection='3d')
    ax_estimate.scatter(x_test[:,0], x_test[:,1], p_test, color='b')
    ax_estimate.scatter(x_test[:,0], x_test[:,1], p_test+sigma_test, color='g')
    ax_estimate.scatter(x_test[:,0], x_test[:,1], p_test-sigma_test, color='r')
    plt.show()

