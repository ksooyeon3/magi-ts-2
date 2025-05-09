import torch
import gpytorch
import numpy as np
from .kernels.matern import MaternKernel
from .kernels.grid_interpolation import GridInterpolationKernel
from .integrator import RungeKutta

torch.set_default_dtype(torch.double)

class KISSGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid, interpolation_orders):
        super(KISSGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GridInterpolationKernel(
                MaternKernel(), grid=grid, interpolation_orders=interpolation_orders
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FMAGI(object):
    def __init__(self, ys, dynamic, grid_size=201, interpolation_orders=3):
        self.grid_size = grid_size
        self.comp_size = len(ys)
        for i in range(self.comp_size):
            if (not torch.is_tensor(ys[i])):
                ys[i] = torch.tensor(ys[i]).squeeze()
        self.ys = ys
        self.fOde = dynamic
        self._kiss_gp_initialization(interpolation_orders=interpolation_orders)

    def _kiss_gp_initialization(self, interpolation_orders=3, training_iterations=100):
        tmin = self.ys[0][:,0].min()
        tmax = self.ys[0][:,0].max()
        for i in range(1, self.comp_size):
            tmin = torch.min(tmin, self.ys[i][:,0].min())
            tmax = torch.max(tmax, self.ys[i][:,0].max())
        spacing = (tmax - tmin) / (self.grid_size - 1)
        padding = int((interpolation_orders + 1) / 2)
        grid_bounds = (tmin - padding * spacing, tmax + padding * spacing)
        self.grid = torch.linspace(grid_bounds[0], grid_bounds[1], self.grid_size+2*padding)
        self.gp_models = []
        for i in range(self.comp_size):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            ti = self.ys[i][:,0]
            yi = self.ys[i][:,1]
            ti = ti[~torch.isnan(yi)]
            yi = yi[~torch.isnan(yi)]
            model = KISSGPRegressionModel(ti, yi, likelihood, self.grid, interpolation_orders)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # loss function
            for j in range(training_iterations):
                optimizer.zero_grad()
                output = model(ti)
                loss = -mll(output, yi)
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            self.gp_models.append(model)
        self.grid = self.grid[padding:-padding] # remove extended grid points

    def map(self, max_epoch=1000, 
            learning_rate=1e-3, decay_learning_rate=True,
            hyperparams_update=True, dynamic_standardization=False,
            verbose=False, returnX=False):
        gpmat = []
        u = torch.empty(self.grid_size, self.comp_size)
        x = torch.empty(self.grid_size, self.comp_size)
        dxdtGP = torch.empty(self.grid_size, self.comp_size)
        with torch.no_grad():
            for i in range(self.comp_size):
                ti = self.ys[i][:,0]
                model = self.gp_models[i]
                mean = model.mean_module.constant.item()
                outputscale = model.covar_module.outputscale.item()
                noisescale = model.likelihood.noise.item()
                nugget = noisescale / outputscale
                grid_kernel = model.covar_module.base_kernel
                base_kernel = grid_kernel.base_kernel
                # compute mean for grid points
                xi = model(self.grid).mean
                LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                ui = LC.solve_triangular(xi-mean, upper=False) / np.sqrt(outputscale)
                # compute gradient for grid points
                m = LC.solve_triangular(base_kernel.dCdx2(self.grid,self.grid).to_dense(), upper=False).t()
                dxi = m.matmul(ui) * np.sqrt(outputscale)
                LK = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky()
                # compute covariance for x|grid
                s = LC.solve_triangular(grid_kernel(self.grid,ti).to_dense(), upper=False).t()
                LQ = (grid_kernel(ti,ti).add_jitter(nugget) - s.matmul(s.t())).add_jitter(1e-6)._cholesky()
                # store information
                u[:,i] = ui
                x[:,i] = xi
                dxdtGP[:,i] = dxi
                gpmat.append({'LC':LC,'m':m,'LK':LK,'s':s,'LQ':LQ})

        # output standardization for fOde
        if (dynamic_standardization):
            dxdtGP_means = torch.mean(dxdtGP, axis=0)
            dxdtGP_stds = torch.std(dxdtGP, axis=0)
            self.fOde.update_output_layer(dxdtGP_means, dxdtGP_stds)

        # optimizer for u and theta
        state_optimizer = torch.optim.Adam([u], lr=learning_rate)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr=learning_rate)
        theta_lambda = lambda epoch: (epoch+1) ** (-0.5)
        theta_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(theta_optimizer, lr_lambda=theta_lambda)

        # optimize initial theta
        # attach gradient for theta
        self.fOde.train()
        for param in self.fOde.parameters():
            param.requires_grad_(True)
        for _ in range(200):
            dxdtOde = self.fOde(x)
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.comp_size)
            for i in range(self.comp_size):
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                dxidtGP = gpmat[i]['m'].matmul(u[:,i]) * np.sqrt(outputscale)
                dxidtError = gpmat[i]['LK'].solve_triangular(dxdtOde[:,i]-dxidtGP, upper=False)
                lkh[i] = -0.5/outputscale * dxidtError.square().mean()
            theta_loss = -torch.sum(lkh)
            theta_loss.backward()
            theta_optimizer.step()
        # detach theta gradient
        self.fOde.eval()
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        for epoch in range(max_epoch):
            # optimize u (x after Cholesky decomposition)
            u.requires_grad_(True)
            for _ in range(1):
                state_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty_like(u)
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])
                dxdtOde = self.fOde(x)
                lkh = torch.zeros((self.comp_size, 3))
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * u[:,i].square().sum()
                    # p(Y[I] = y[I] | X[I] = x[I])
                    xiGP = mean + gpmat[i]['s'].matmul(u[:,i]) * np.sqrt(outputscale)
                    yiError = gpmat[i]['LQ'].solve_triangular(torch.nan_to_num(self.ys[i][:,1]-xiGP, nan=0.0), upper=False)
                    lkh[i,1] = -0.5/outputscale * yiError.square().sum() 
                    lkh[i,1] = lkh[i,1] * self.grid_size / torch.sum(~torch.isnan(self.ys[i][:,1])).item()
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                    dxidtGP = gpmat[i]['m'].matmul(u[:,i]) * np.sqrt(outputscale)
                    dxidtError = gpmat[i]['LK'].solve_triangular(dxdtOde[:,i]-dxidtGP, upper=False)
                    lkh[i,2] = -0.5/outputscale * dxidtError.square().sum() 
                    # lkh[i,2] = lkh[i,2] / self.grid_size * torch.sum(~torch.isnan(self.ys[i][:,1])).item()
                state_loss = -torch.sum(lkh)  / self.grid_size
                state_loss.backward()
                state_optimizer.step()
            # detach gradient information
            u.requires_grad_(False)

            if (verbose and (epoch==0 or (epoch+1) % int(max_epoch/5) == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,max_epoch,state_loss.item()))

            # reconstruct x
            x = torch.empty_like(u)
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])

            if ((epoch+1) < max_epoch):
                # update hyperparameter
                if (((epoch+1) % int(max_epoch/5) == 0) and hyperparams_update):
                    dxdtOde = self.fOde(x)
                    for i in range(self.comp_size):
                        ti = self.ys[i][:,0]
                        yi = self.ys[i][:,1]
                        xi = x[:,i]
                        model = self.gp_models[i]
                        model.train() 
                        model.likelihood.train()
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                        for _ in range(5):
                            optimizer.zero_grad()
                            mean = model.mean_module.constant.item()
                            outputscale = model.covar_module.outputscale.item()
                            noisescale = model.likelihood.noise.item()
                            nugget = noisescale / outputscale
                            grid_kernel = model.covar_module.base_kernel
                            base_kernel = grid_kernel.base_kernel
                            # p(X[I] = x[I])
                            LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky() 
                            ui = LC.solve_triangular(xi-mean, upper=False) / np.sqrt(outputscale)
                            lkh1 = -0.5 * ui.square().sum()
                            lkh1 = lkh1 - 0.5 * self.grid_size * np.log(outputscale) - LC.logdet()
                            # p(Y[I] = y[I] | X[I] = x[I])
                            s = LC.solve_triangular(grid_kernel(self.grid,ti).to_dense(), upper=False).t()
                            LQ = (grid_kernel(ti,ti).add_jitter(nugget) - s.matmul(s.t())).add_jitter(1e-6)._cholesky()
                            xiGP = mean + s.matmul(ui) * np.sqrt(outputscale)
                            yiError = LQ.solve_triangular(torch.nan_to_num(yi-xiGP, nan=0.0), upper=False)
                            lkh2 = -0.5/outputscale * yiError.square().sum()
                            lkh2 = lkh2 - 0.5 * self.grid_size * np.log(outputscale) - LQ.logdet()
                            lkh2 = lkh2 * self.grid_size / torch.sum(~torch.isnan(yi)).item()
                            # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                            m = LC.solve_triangular(base_kernel.dCdx2(self.grid,self.grid).to_dense(), upper=False).t()
                            LK = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky()
                            dxidtGP = m.matmul(ui) * np.sqrt(outputscale)
                            dxidtError = LK.solve_triangular(dxdtOde[:,i]-dxidtGP, upper=False)
                            lkh3 = - 0.5 / outputscale * dxidtError.square().sum()
                            lkh3 = lkh3 - 0.5 * self.grid_size * np.log(outputscale) - LK.logdet()
                            # lkh3 = lkh3 / self.grid_size * torch.sum(~torch.isnan(yi)).item()
                            loss = -(lkh1 + lkh2 + lkh3) / self.grid_size
                            loss.backward()
                            optimizer.step()
                        model.eval()
                        model.likelihood.eval()
                        self.gp_models[i] = model
                        # update gpmat
                        with torch.no_grad():
                            mean = model.mean_module.constant.item()
                            outputscale = model.covar_module.outputscale.item()
                            noisescale = model.likelihood.noise.item()
                            nugget = noisescale / outputscale
                            grid_kernel = model.covar_module.base_kernel
                            base_kernel = grid_kernel.base_kernel
                            # compute mean for grid points
                            LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                            ui = LC.solve_triangular(xi-mean, upper=False) / np.sqrt(outputscale)
                            # compute gradient for grid points
                            m = LC.solve_triangular(base_kernel.dCdx2(self.grid,self.grid).to_dense(), upper=False).t()
                            LK = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky()
                            # compute covariance for x|grid
                            s = LC.solve_triangular(grid_kernel(self.grid,ti).to_dense(), upper=False).t()
                            LQ = (grid_kernel(ti,ti).add_jitter(nugget) - s.matmul(s.t())).add_jitter(1e-6)._cholesky()
                            # store information
                            u[:,i] = ui
                            gpmat[i] = {'LC':LC,'m':m,'LK':LK,'s':s,'LQ':LQ}

                self.fOde.train()
                for param in self.fOde.parameters():
                    param.requires_grad_(True)
                for _ in range(1):
                    dxdtOde = self.fOde(x)
                    theta_optimizer.zero_grad()
                    lkh = torch.zeros(self.comp_size)
                    for i in range(self.comp_size):
                        outputscale = self.gp_models[i].covar_module.outputscale.item()
                        dxidtGP = gpmat[i]['m'].matmul(u[:,i]) * np.sqrt(outputscale)
                        dxidtError = gpmat[i]['LK'].solve_triangular(dxdtOde[:,i]-dxidtGP, upper=False)
                        lkh[i] = -0.5/outputscale * dxidtError.square().mean()
                    theta_loss = -torch.sum(lkh)
                    theta_loss.backward()
                    theta_optimizer.step()
                if (decay_learning_rate):
                    theta_lr_scheduler.step()
                # detach theta gradient
                self.fOde.eval()
                for param in self.fOde.parameters():
                    param.requires_grad_(False)

        if (returnX):
            return (self.grid.numpy(), x.numpy())

    def predict(self, tp, t0, x0, random=False, **params):
        # obtain prediction by ode numerical integration / gaussian process map
        if (random):
            self.fOde.train() # training mode with dropout
        else:
            self.fOde.eval() # evaluation mode without dropout
            for param in self.fOde.parameters():
                param.requires_grad_(False)
        # call integrater
        itg = RungeKutta(self.fOde)
        # preprocess input
        if (not torch.is_tensor(t0)):
            t0 = torch.tensor(t0).reshape(-1,)
        if (not torch.is_tensor(tp)):
            tp = torch.tensor(tp).reshape(-1,)
        if (not torch.is_tensor(x0)):
            x0 = torch.tensor(x0).squeeze()
            if (x0.ndimension()==1):
                x0 = x0.unsqueeze(0)
        xp = itg.forward(x0[-1,:], torch.cat((t0[-1].unsqueeze(0),tp)), **params)[1:,:]      
        t = torch.cat((t0,tp))
        x = torch.cat((x0,xp))
        return (t.numpy(), x.numpy())
