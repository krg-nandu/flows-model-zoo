import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import cINN

def evalPosterior(model, c_mu, c_var, n_samples, n_feat_in, n_eval_batch=1024, n_ex=0):
    x_eval = torch.zeros(n_eval_batch, n_samples, n_feat_in)
    for k in range(n_eval_batch):
        x_eval[k,:,:] = torch.distributions.MultivariateNormal(c_mu, torch.diag(c_var)).sample((n_samples,))
    z = torch.empty(n_eval_batch,n_feat_in).normal_(mean=0, std=1)
    eval_output = model.inverse([x_eval, z]).detach()

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.hist(eval_output[:,0], 50, alpha=0.3, color='r', density=True)
    ymin, ymax = ax.get_ylim()
    ax.plot([c_mu[0], c_mu[0]], [ymin, ymax], '--', c='r')
    ax.set_title('mu_1')

    ax = fig.add_subplot(122)
    ax.hist(eval_output[:,1], 50, alpha=0.3, color='g', density=True)
    ax.plot([c_mu[1], c_mu[1]], [ymin, ymax], '--', c='g')
    ax.set_title('mu_2')
    plt.savefig('results/multi_var_gaussian_means_ex_{}.png'.format(n_ex))
    plt.close()


def bayesFlowExample(is_train=False):
    #import ipdb; ipdb.set_trace()
    summary_dim = 5
    n_feat_in = 2

    ### Univariate gaussian: paramters from uniform prior
    #mu = torch.distributions.uniform.Uniform(torch.tensor([-1.]), torch.tensor([1.]))
    #var = torch.distributions.uniform.Uniform(torch.tensor([0.1]), torch.tensor([0.5])) 
    ##samplesize = torch.distributions.categorical.Categorical(torch.tensor([1/1000.]*1000))

    ### Multivariate gaussian: parameters from normal prior
    var = torch.tensor([1., 1.])
    mu = torch.distributions.MultivariateNormal(torch.tensor([-5., 3.]), torch.diag(var))
    
    epochs = 8096
    n_batch = 1024
    n_samples = 256
    
    model = cINN(net_params=[n_feat_in, 50, 20, 50, summary_dim, True], 
                 n_blocks = 5, 
                 data_dim = n_feat_in, 
                 summary_dim = summary_dim,
                 batch_size = n_batch)
    save_path = 'models/cINN_epochs_{}_nbatch_{}_nsamples_{}_mulvarGaussian.pth'.format(epochs,n_batch,n_samples)

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

            # for the univariate example
            #c_var = var.sample((n_batch,))
            #label = torch.cat([c_mu, c_var], dim=1)

            label = c_mu #NOTE this is a hack for now!

            x_train = torch.zeros(n_batch, n_samples, n_feat_in)
            for k in range(n_batch):
                #x_train[k].normal_(mean=float(c_mu[k]), std=float(c_var[k]))
                x_train[k,:,:] = torch.distributions.MultivariateNormal(c_mu[k], torch.diag(var)).sample((n_samples,))

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
        
        torch.save(model.state_dict(), save_path)    
    else:
        print('Evaluating model: ')
        model.load_state_dict(torch.load(save_path))

    model.eval()

    for k in range(10):
        c_mu = mu.sample()
        evalPosterior(model, c_mu=c_mu, c_var=var, n_samples=n_samples, n_feat_in=n_feat_in, n_eval_batch=1024, n_ex=k)
    #import ipdb; ipdb.set_trace()

def main():
    bayesFlowExample()

if __name__ =='__main__':
    main()
