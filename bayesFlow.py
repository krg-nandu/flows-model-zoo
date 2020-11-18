import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import cINN

'''
class AffineCouplingBlock(torch.nn.Module):
    def __init__(self, n_half, summary_dim):
        super(AffineCouplingBlock, self).__init__()
        self.t1 = Feedforward(n_half+summary_dim, 50, n_half, rectification='ELU')
        self.t2 = Feedforward(n_half+summary_dim, 50, n_half, rectification='ELU')
        self.s1 = Feedforward(n_half+summary_dim, 50, n_half, rectification='ELU')
        self.s2 = Feedforward(n_half+summary_dim, 50, n_half, rectification='ELU')
        self.log_det = 0

    def forward(self, U):
        #import ipdb; ipdb.set_trace()
        self.log_det = 0
        D = np.int(np.floor(U[1].shape[-1]/2.))
        u_1 = U[1][..., :D]
        u_2 = U[1][..., D:]
        
        log_det_1 = self.s1(torch.cat([U[0], u_2]))
        v_1 = torch.mul(u_1, torch.exp(log_det_1)) + self.t1(torch.cat([U[0], u_2]))

        log_det_2 = self.s2(torch.cat([U[0], v_1]))
        v_2 = torch.mul(u_2, torch.exp(log_det_2)) + self.t2(torch.cat([U[0], v_1]))
        V = torch.cat([v_1, v_2], axis=-1)

        self.log_det = log_det_1 + log_det_2

        return V

    def inverse(self, V):
        D = np.int(np.floor(V[1].shape[-1]/2.))
        v_1 = V[1][..., :D]
        v_2 = V[1][..., D:]
        u_2 = torch.mul(v_2 - self.t2(torch.cat([V[0], v_1])), torch.exp(-self.s2(torch.cat([V[0], v_1]))))
        u_1 = torch.mul(v_1 - self.t1(torch.cat([V[0], u_2])), torch.exp(-self.s1(torch.cat([V[0], u_2]))))
        U = torch.cat([u_2, u_1], axis=-1)
        return U

class cINN(torch.nn.Module):
    def __init__(self, net_params, n_blocks, n_half, summary_dim):
        super(cINN, self).__init__()
        self.summary_net = PermInvariantNetwork(*net_params)
        self.n_blocks = n_blocks
        self.blocks = []
        self.log_dets = torch.tensor(0.) 

        for k in range(self.n_blocks):
            self.blocks.append(AffineCouplingBlock(n_half, summary_dim))
            setattr(self, 'block_{}'.format(k), self.blocks[-1])

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        x_hat = self.summary_net(x[0])
        y = x[1]
        self.log_dets = torch.tensor(0.)
        for k in range(self.n_blocks):
            y = self.blocks[k]([x_hat, y])
            self.log_dets += self.blocks[k].log_det[0]
        return y

    def inverse(self, y):
        y_hat = self.summary_net(y[0])
        x = y[1]
        for k in range(self.n_blocks-1,-1,-1):
            x = self.blocks[k].inverse([y_hat, x])
        return x

    def computeLoss(self, pred):
        loss = torch.mean((torch.norm(pred)**2)/2. - torch.sum(self.log_dets))
        return loss
'''

def bayesFlowExample(is_train=True):
    #import ipdb; ipdb.set_trace()
    summary_dim = 5
    n_feat_in = 2

    mu = torch.distributions.uniform.Uniform(torch.tensor([-1.]), torch.tensor([1.]))
    var = torch.distributions.uniform.Uniform(torch.tensor([0.1]), torch.tensor([0.5]))
    
    #samplesize = torch.distributions.categorical.Categorical(torch.tensor([1/1000.]*1000))
    
    epochs = 8096
    n_batch = 1024
    n_samples = 256
    
    model = cINN(net_params=[n_feat_in, 50, 20, 50, summary_dim, True], 
                 n_blocks = 5, 
                 data_dim = n_feat_in, 
                 summary_dim = summary_dim,
                 batch_size = n_batch)
    save_path = 'models/cINN_epochs_{}_nbatch_{}_nsamples_{}_univarGaussian.pth'.format(epochs,n_batch,n_samples)

    if is_train:
        criterion = model.computeLoss
        optimizer = torch.optim.Adam(list(model.parameters()), lr = 1e-3)

        model.train()
        print('Training for {} epochs...'.format(epochs))

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for epoch in range(epochs):

            optimizer.zero_grad()
            c_mu = mu.sample((n_batch,))
            c_var = var.sample((n_batch,))
            label = torch.cat([c_mu, c_var], dim=1)
            #label = c_mu #NOTE this is a hack for now!

            x_train = torch.zeros(n_batch, n_samples, n_feat_in)
            for k in range(n_batch):
                x_train[k].normal_(mean=float(c_mu[k]), std=float(c_var[k]))
                #x_train[k,:,:] = torch.distributions.MultivariateNormal()

            phi_output = model([x_train, label])
            loss = criterion(phi_output)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                op = phi_output.detach()
                ax1.clear()
                ax2.clear()
                #ax1.scatter(label[:,0], label[:,1], alpha=0.2)
                ax1.hist(label[:,0], alpha=0.2, color='r')
                ax1.hist(label[:,1], alpha=0.2, color='g')
                ax1.set_title('theta')

                #ax2.scatter(op[:,0], op[:,1], alpha=0.2)
                ax2.hist(op[:,0], alpha=0.2, color='r')
                ax2.hist(op[:,1], alpha=0.2, color='g')
                ax2.set_title('z')
                plt.pause(1)
 
            print('Epoch {}: train loss: {}'.format(epoch, float(loss)))
        
        torch.save(model.state_dict(), 'cINN_e8096_multivariate_mu.pth')    
    else:
        print('Evaluating model: ')
        model.load_state_dict(torch.load(save_path))

    model.eval()
    import os; os._exit(0)

    mu_gt, mu_pred = [], []
    var_gt, var_pred = [], []
    sizes = []
    for eval_run in tqdm.tqdm(range(256)):
        c_mu = mu.sample()
        c_var = var.sample()
        #ss = 100 + samplesize.sample().item()
        ss = 50

        Z = torch.distributions.MultivariateNormal(c_mu, scale_tril=torch.diag(c_var))
        #x_train = torch.empty(ss, 1).normal_(mean=c_mu[0], std=c_var[0])
        x_train = Z.sample((ss,))
 
        #x_train = torch.empty(ss, 1).normal_(mean=c_mu[0], std=c_var[0])

        mu_gt.append(c_mu[0].item())
        var_gt.append(c_mu[1].item()) #var_gt.append(c_var[0].item())
        sizes.append(((ss - 100)/1000.) * 255.)

        #x_summary = model.summary_net(x_train)
        mu_preds, var_preds = [], []
        for k in range(100):
            z = torch.empty(2,).normal_(mean=0, std=1)
            phi_outputs = model.inverse([x_train, z])
            mu_preds.append(phi_outputs[0].item())
            var_preds.append(phi_outputs[1].item())

        mu_pred.append(np.mean(mu_preds))
        var_pred.append(np.mean(var_preds))

        '''       
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(mu_preds, var_preds, alpha=0.5)
        ax.scatter(mu_gt[-1], var_gt[-1], c='r')
        plt.savefig('ex_{}.png'.format(eval_run))
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        '''

    #import ipdb; ipdb.set_trace()  
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(mu_gt, mu_pred, 3,  cmap='Greys')
    ax.set_title('mu')

    ax = fig.add_subplot(122)
    ax.scatter(var_gt, var_pred, 3,  cmap='Greys')
    ax.set_title('var')
    plt.show()

def main():
    bayesFlowExample()

if __name__ =='__main__':
    main()
