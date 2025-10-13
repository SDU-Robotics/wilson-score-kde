import torch
import numpy as np

class WSKDE:
    def __init__(self, H: torch.Tensor | np.ndarray) -> None:
        '''
            H:  The bandwidth matrix which must be symmetric and positive semi-definite
            Note that the bandwidth kernel is suggested to be proportional to the squareroot
            of the covariance of the data. In the case of a Gaussian kernel this would mean
            H = Sigma^(1/2). Note that this relationship explains why the equations using H
            looks different from the standard multivariate Gaussian which uses Sigma.
            
        '''
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H)

        assert len(H)>0
        assert H.shape[0] == H.shape[1] #check symmetry
        torch.linalg.cholesky(H) #check if positive-definite (cholesky will fail if it is not)
        self.H = H
        self.H_inv = torch.linalg.inv(H)
        self.d = H.shape[1]
        self.n = 0
        self.det_H = torch.linalg.det(H)

        self.k22 = 1.0/( pow(2.0, self.d) * pow(pow(torch.pi,self.d), 0.5) ) #See article appendix
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.precomputed_K_hxi = None
        self.precomputed_sum_K_hxi = None
        self.precomputed_mh = None
        self.precomputed_nh = None
    
    def ws_kde(self, x:torch.Tensor | np.ndarray, z:float=1.96) -> tuple[torch.Tensor, torch.Tensor]:
        '''
            This is the way to call the wilson score function. This will clear the cached equation values
            so they are recomputed for new x-points and/or z-values.

            x: Batch of B d-dimensional quary points {B x d}
            z: float
            return: tuple p and sigma, both shape {B}
        '''

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        self.precomputed_K_hxi = None
        self.precomputed_sum_K_hxi = None
        self.precomputed_mh = None
        self.precomputed_nh = None
        return (self.p_wskde(x,z),self.sigma_wskde(x,z))
    
    def p_wskde(self, x : torch.Tensor, z : float) -> torch.Tensor:
        ''' 
            Internal method for computing the p_wskde value.
            Warning: It will use cached variables, so if called directly (instead of through ws_kde)
            the output will depend on earlier computations

            x: {B x d}
            z: float
            return: {B}
        '''
        res = 1.0/(self.n_h(x) + pow(z,2)) * (self.n_h(x)*self.m_h(x)+pow(z,2.0)/2.0)
        return res

    def n_h(self, x : torch.Tensor) -> torch.Tensor: 
        ''' 
            x: {B x d}
            return: {B}
        '''
        if self.precomputed_nh is None:
            self.precomputed_nh = self.det_H / self.k22 * self.sum_K_hxi(x)
        return self.precomputed_nh
    
    def sum_K_hxi(self, x: torch.Tensor) -> torch.Tensor:
        if self.precomputed_sum_K_hxi is None:
            self.precomputed_sum_K_hxi = torch.sum(self.K_hxi(x),1)
        return self.precomputed_sum_K_hxi


    def K_hxi(self, x: torch.Tensor) -> torch.Tensor:
        ''' 
            The method computes the distance from all B quary points to to each 
            of the n training points, and inserts each result into the kernel. The
            output of the Kernel is a single float. This explains why the input has
            dimensionality {B x d} while the output has dimensionality {B x n}

            x: {B x d}
            return: {B x n}
        '''
        if self.precomputed_K_hxi is None:
            # Compute the distance to all training points for each point in batch using broadcasting:
            # {B x 1 x d} - {1 x n x d}) => {B x n x d}
            delta_xi = torch.reshape(x,(-1,1,self.d)) - torch.reshape(self.X, (1,-1,self.d))

            # manual reshape to (1,1,d,d) and (B,n,d,1), to allow broadcast multiplication
            # of the inverse bandwidth kernel. ui: {B,n,d,1}
            ui = torch.reshape(self.H_inv,(1,1,self.d,self.d)) @ delta_xi.unsqueeze(-1)

            # Multiply with inverse det(H) (recall that H=Sigma^0.5, leading to the standard change
            # of variables requiring multiplication with det(Sigma^-0.5)= 1.0/det(H)
            self.precomputed_K_hxi =  1.0/self.det_H * self.K_i(ui)
        
        return self.precomputed_K_hxi

    def K_i(self, ui: torch.Tensor) -> torch.Tensor:
        ''' 
            PDF for a d-dimensional standard Normal distribution
            ui: {B x n x d x 1}
            return: {B x n}
        '''
        # {B x n x 1 x d} @ {B x n x d x 1} => {B x n x 1 x 1} ==squeeze((-1,-2))==> {B x n}
        res = pow(2.0*torch.pi,-self.d/2.0)*torch.exp(-0.5*torch.transpose(ui,-2,-1)@ui)
        return res.squeeze((-1,-2))
    
    def m_h(self,x : torch.Tensor) -> torch.Tensor:
        ''' 
            Note: Although theoretically the density is greater than zero far from the center
            of a Gaussian kernel, numerical in-precision can cause it to be zero. This can lead
            to zero-division. In these cases m_h is defined to be 0.5.

            x: {B x d}
            return: {B}
        '''
        if self.precomputed_mh is None:
            # the shape of self.K_hxi(x) is {B x n}
            # the shape of self.K_hxi_sum(x) is {B}
            # The Y_i value corresponding to the X_i datapoint is needed for each B quaries,
            # so self.K_hxi(x) with shape {B x n } => Y is shape { 1 x n}, and the sum is
            # taken over the n samples => res is {B}
        
            self.precomputed_mh = torch.sum(self.K_hxi(x) * torch.reshape(self.Y,(1,-1)) , 1) / self.sum_K_hxi(x)
            self.precomputed_mh[self.sum_K_hxi(x) == 0] = 0.5
        return self.precomputed_mh

    def sigma_wskde(self,x : torch.Tensor, z : float) -> torch.Tensor:
        '''
            Internal method for computing the sigma_wskde value.
            Warning: It will use cached variables, so if called directly (instead of through ws_kde)
            the output will depend on earlier computations
        
            x: {B x d}
            return: {B}
        '''
        fraction = z/(self.n_h(x) + pow(z,2.0)) # shape B
        mh_x = self.m_h(x)
        inside_sqrt = self.n_h(x)*mh_x*(1-mh_x)+pow(z,2.0)/4.0 # shape B
        res = fraction*torch.pow(inside_sqrt,0.5) #shape B
        return res

    def set_training_samples(self, X : torch.Tensor | np.ndarray, Y : torch.Tensor | np.ndarray) -> None:
        '''
            shape of tensors:
            x: {n x 2}
            y: {n}
        '''
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(Y, np.ndarray):   
            Y = torch.from_numpy(Y)

        self.X = X #shape (n,2)
        self.Y = Y #shape (n)
        self.n = len(X)

    def p_na(self, x : torch.Tensor, z : float) -> torch.Tensor:
        ''' 
            Internal method for computing the p_na value.
            Warning: It will use cached variables, so if called directly (instead of through na_kde)
            the output will depend on earlier computations

            x: {B x d}
            return {B}       
        '''
        return self.m_h(x)
    
    def sigma_na(self,x : torch.Tensor, z : float) -> torch.Tensor:
        '''
            Internal method for computing the sigma_na value.

            Note: Although theoretically the density is greater than zero far from the center
            of a Gaussian kernel, numerical in-precision can cause it to be zero. This can lead
            to zero-division. In these cases m_h in the equation is defined to be 0.5.

            Warning: It will use cached variables, so if called directly (instead of through na_kde)
            the output will depend on earlier computations
        
            x: {B x d}
            z: float
            return: {B}
        '''

        #Reshape so K_hxi: {B,n}, Y {n}, m_h{B} all become {B,n}
        sample_variance = torch.sum(
            self.K_hxi(x) * torch.pow(
                torch.reshape(self.Y,(1,-1)) - torch.reshape(self.m_h(x),(-1,1)),2), 1)/ self.sum_K_hxi(x)
        numerator = self.k22*sample_variance
        denominator = self.det_H*self.sum_K_hxi(x)
        res = z*torch.pow(numerator/denominator,0.5)
        res[self.sum_K_hxi(x) == 0] = 0.5
        return res

    def na_kde(self, x:torch.Tensor, z:float=1.96) -> tuple[torch.Tensor, torch.Tensor]:
        '''
            This is the way to call the normal approximation KDE function. This will clear the
            cached equation values so they are recomputed for new x-points and/or z-values.

            Note: The normal approximation KDE equations (na_kde, p_kde, and sigma_kde) are
            not a part of the Wilson Score equations. They are keps as part a part of the class
            to allow for easy comparison with normal kernel density estimation 

            x: {B x d}
            z: float
            return: {B}
        '''
        self.precomputed_K_hxi = None
        self.precomputed_sum_K_hxi = None
        self.precomputed_mh = None
        self.precomputed_nh = None
        return (self.p_na(x,z),self.sigma_na(x,z))
    
    def __call__(self, x:torch.Tensor, kde_method="ws_kde", z:float=1.96) -> tuple[torch.Tensor, torch.Tensor]:
        '''
            The call-method is used to call either the Wilson Score KDE or the
            Normal Approximation KDE 

            x: Batch of B d-dimensional quary points {B x d}
            z: float
            return: tuple p and sigma, both shape {B}
        ''' 
        if kde_method=="ws_kde":
            return self.ws_kde(x,z)
        elif kde_method=="na_kde":
            return self.na_kde(x,z)
        else:
            raise ValueError("Invalid method. Use 'ws_kde' or 'na_kde'")
