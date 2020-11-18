import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import INN

def SimpleFlowExample():
    data_dim = 2

    #mu = torch.distributions.uniform.Uniform(torch.tensor([-1., -1.]), torch.tensor([1., 1.]))
    mu = torch.distributions.MultivariateNormal(torch.tensor([-5.,-2.]), scale_tril=torch.diag(torch.tensor([1.,1.])))
    #var = torch.distributions.uniform.Uniform(torch.tensor([0.1, 0.1]), torch.tensor([0.1, 0.1]))
    #samplesize = torch.distributions.categorical.Categorical(torch.tensor([1/1000.]*1000))
    
    epochs = 2048
    batch_size = 4096

    model = INN(n_blocks=5, data_dim=data_dim, batch_size=batch_size)
    criterion = model.computeLoss
    optimizer = torch.optim.Adam(list(model.parameters()), lr = 1e-3)
    model.train()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    print('Training for {} epochs...'.format(epochs))
    for epoch in range(epochs):
        loss_sum = 0
        optimizer.zero_grad()
        x_train = mu.sample((batch_size,))
        output = model(x_train)

        loss = criterion(output)
        loss.backward()
        optimizer.step()
        print('Epoch {}: train loss: {}'.format(epoch, float(loss)))    # Backward pass

        if epoch % 250 == 0:
            ax1.hist(x_train[:,0])
            ax2.hist(output.detach()[:,0])
            plt.pause(1)
    torch.save(model.state_dict(), 'simple_flow_demo.pth')

def SimpleFlowInverse():
    model = INN(n_blocks=5, data_dim=2, batch_size=4096)
    model.load_state_dict(torch.load('simple_flow_demo.pth'))
    z = torch.distributions.MultivariateNormal(torch.tensor([0.,0.]), scale_tril=torch.diag(torch.tensor([1.,1.])))
    z_samples = z.sample((4096,))
    x_samples = model.inverse(z_samples)

    x_data = x_samples.detach()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(x_data[:,0], x_data[:,1], alpha=0.2)
    ax2.scatter(z_samples[:,0], z_samples[:,1], alpha=0.2)

    plt.show()    

    mu = torch.distributions.uniform.Uniform(torch.tensor([-1., -1.]), torch.tensor([1., 1.]))
    x_samples = mu.sample((4096,))
    z_samples = model(x_samples)

    z_data = z_samples.detach()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(x_samples[:,0], x_samples[:,1], alpha=0.2)
    ax2.scatter(z_data[:,0], z_data[:,1], alpha=0.2)

    plt.show()    



def main():
    SimpleFlowExample()
    #SimpleFlowInverse()

if __name__ =='__main__':
    main()
