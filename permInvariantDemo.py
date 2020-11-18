import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import PermInvariantNetwork

def univariateGaussianExample():
    model = PermInvariantNetwork(n_features_in = 1, 
                              n_hidden_psi = 10, 
                              n_output_psi = 5, 
                              n_hidden_phi = 10,
                              n_features_out = 2,
                              is_batched = False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr = 1e-3)

    model.train()
    epochs = 256
    print('Training for {} epochs...'.format(epochs))

    mu = torch.distributions.uniform.Uniform(torch.tensor([-5.]), torch.tensor([5]))
    var = torch.distributions.uniform.Uniform(torch.tensor([0.2]), torch.tensor([2]))
    samplesize = torch.distributions.categorical.Categorical(torch.tensor([1/1000.]*1000))

    for epoch in range(epochs):
        ss = 100 + samplesize.sample().item()
 
        for k in range(100):
            optimizer.zero_grad()
            c_mu = mu.sample()
            c_var = var.sample()
            label = torch.cat([c_mu, c_var])

            # generate a batch
            x_train = torch.empty(ss, 1).normal_(mean=c_mu[0], std=c_var[0])
            phi_output = model(x_train)

            loss = criterion(phi_output, label)
            loss.backward()
            optimizer.step()

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
    
    print('Evaluating model: ')

    model.eval()

    mu_gt, mu_pred = [], []
    var_gt, var_pred = [], []
    sizes = []
    for eval_run in range(1000):
        c_mu = mu.sample()
        c_var = var.sample()
        ss = 100 + samplesize.sample().item()

        x_train = torch.empty(ss, 1).normal_(mean=c_mu[0], std=c_var[0])

        mu_gt.append(c_mu[0].item())
        var_gt.append(c_var[0].item())
        sizes.append(((ss - 100)/1000.) * 255.)

        phi_outputs = model(x_train)
        mu_pred.append(phi_outputs[0].item())
        var_pred.append(phi_outputs[1].item())

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(mu_gt, mu_pred, 3, c=sizes, cmap='Greys')
    ax.set_title('mu')

    ax = fig.add_subplot(122)
    ax.scatter(var_gt, var_pred, 3, c=sizes, cmap='Greys')
    ax.set_title('var')
    plt.show()

def univariateGaussianExampleBatched():
    n_feat_in = 1
    model = PermInvariantNetwork(n_features_in = n_feat_in, 
                              n_hidden_psi = 10, 
                              n_output_psi = 5, 
                              n_hidden_phi = 10,
                              n_features_out = 2)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr = 1e-2)

    model.train()
    epochs = 512
    print('Training for {} epochs...'.format(epochs))

    mu = torch.distributions.uniform.Uniform(torch.tensor([-5.]), torch.tensor([5]))
    var = torch.distributions.uniform.Uniform(torch.tensor([0.2]), torch.tensor([2]))
    samplesize = torch.distributions.categorical.Categorical(torch.tensor([1/1000.]*1000))

    n_batch = 128

    for epoch in range(epochs):
        ss = 100 + samplesize.sample().item()
        optimizer.zero_grad()

        c_mu = mu.sample((n_batch,))
        c_var = var.sample((n_batch,))
        label = torch.cat([c_mu, c_var], dim=1)

        x_train = torch.zeros(n_batch, ss, n_feat_in)
        for k in range(n_batch):
            x_train[k].normal_(mean=float(c_mu[k]), std=float(c_var[k]))
        phi_output = model(x_train)

        loss = criterion(phi_output, label)
        loss.backward()
        optimizer.step()

        print('Epoch {}: train loss: {}'.format(epoch, float(loss)))    # Backward pass
    
    print('Evaluating model: ')

    model.eval()

    n_eval_batch = 1024
    c_mu = mu.sample((n_eval_batch,))
    c_var = var.sample((n_eval_batch,))
    ss = 100 + samplesize.sample().item()
    
    x_eval = torch.zeros(n_eval_batch, ss, n_feat_in)
    for k in range(n_eval_batch):
        x_eval[k].normal_(mean=float(c_mu[k]), std=float(c_var[k]))
    eval_output = model(x_eval).detach()

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(c_mu, eval_output[:,0], 3, cmap='Greys')
    ax.set_title('mu')

    ax = fig.add_subplot(122)
    ax.scatter(c_var, eval_output[:,1], 3, cmap='Greys')
    ax.set_title('var')
    plt.show()

def main():
    #univariateGaussianExample()
    univariateGaussianExampleBatched()

if __name__ =='__main__':
    main()
