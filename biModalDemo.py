import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import INN
import os
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
 
def trainFlow(data_dir):

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_dim = 2
    #mu = torch.distributions.MultivariateNormal(torch.tensor([-5.,-2.]), scale_tril=torch.diag(torch.tensor([1.,1.])))
 
    mix = D.Categorical(torch.ones(2,))
    comp = D.Independent(D.Normal(torch.tensor([[5., 5.],[3., 3.]]), torch.rand(2,2)), 1)
    mu = MixtureSameFamily(mix, comp)

    z = torch.distributions.MultivariateNormal(torch.tensor([0.,0.]), scale_tril=torch.diag(torch.tensor([1.,1.])))
 
    epochs = 2048
    batch_size = 4096*2

    model = INN(n_blocks=10, data_dim=data_dim, batch_size=batch_size)
    criterion = model.computeLoss
    optimizer = torch.optim.Adam(list(model.parameters()), lr = 1e-3)
    model.train()

    fig = plt.figure(figsize=(9,3))
    # data
    ax1 = fig.add_subplot(131)
    # transformed
    ax2 = fig.add_subplot(132)
    # inverse transformed
    ax3 = fig.add_subplot(133)

    print('Training for {} epochs...'.format(epochs))
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_train = mu.sample((batch_size,))
        output = model(x_train)

        loss = criterion(output)
        loss.backward()
        optimizer.step()
        print('Epoch {}: train loss: {}'.format(epoch, float(loss)))    # Backward pass

        if epoch % 10 == 0:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            op = output.detach().numpy()
            ax1.hexbin(x_train[:,0].numpy(), x_train[:, 1].numpy(), gridsize=60, cmap='YlOrRd')
            ax1.set_title('true samples')

            ax2.hexbin(op[:,0], op[:,1], gridsize=60, cmap='Greys')
            ax2.set_title('transformed')

            inv_samples = model.inverse(z.sample((batch_size,)))
            inv_samples = inv_samples.detach().numpy()

            ax3.hexbin(inv_samples[:,0], inv_samples[:, 1], gridsize=60, cmap='YlOrRd')
            ax3.set_title('inverse_transformed')

            plt.savefig(os.path.join(data_dir, 'img%05d.png'%(epoch/10.)))
            plt.pause(1)

    torch.save(model.state_dict(), 'simple_flow_demo.pth')

def main():

    trainFlow(data_dir='bimodal_ex3')

if __name__ =='__main__':
    main()
