import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rectification='ReLU', output_act=False):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act = getattr(torch.nn, rectification)()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.output_act = torch.nn.Tanh()
        self.is_output_act = output_act

    def forward(self, x):
        hidden = self.fc1(x)
        act = self.act(hidden)
        output = self.fc2(act)
        if self.is_output_act:
            output = self.output_act(output)

        return output

class PermInvariantNetwork(torch.nn.Module):
    def __init__(self, n_features_in, n_hidden_psi, n_output_psi, n_hidden_phi, n_features_out, is_batched=True):
        super(PermInvariantNetwork, self).__init__()
        self.f_psi = Feedforward(n_features_in, n_hidden_psi, n_output_psi)
        self.f_omega = Feedforward(n_features_in, n_hidden_psi, n_output_psi)
        self.f_phi = Feedforward(n_output_psi, n_hidden_phi, n_features_out)

        if is_batched:
            self.collapse_dim = 1
        else:
            self.collapse_dim = 0

        self.softmax = torch.nn.Softmax(dim=self.collapse_dim)

    def parameters(self):
        return list(self.f_psi.parameters()) + list(self.f_omega.parameters()) + list(self.f_phi.parameters())

    def ffun(self, x):
        psi_outputs = self.f_psi(x)
        weights = self.softmax(self.f_omega(x))
        phi_inputs = torch.sum(torch.mul(weights, psi_outputs), dim=self.collapse_dim)
        phi_output = self.f_phi(phi_inputs)
        return phi_output

    def forward(self, x):
        y = self.ffun(x)
        return y

class ACBFlow(torch.nn.Module):
    def __init__(self, data_dim):
        super(ACBFlow, self).__init__()
        self.S2T2_input_size = np.int(np.floor(data_dim/2.)) if data_dim%2==0 else np.int(np.floor(data_dim/2.)+1.) 
        self.S1T1_input_size = data_dim - self.S2T2_input_size        

        self.s1 = Feedforward(self.S1T1_input_size, self.S1T1_input_size*5, self.S1T1_input_size, rectification='ELU', output_act=True)
        self.t1 = Feedforward(self.S1T1_input_size, self.S1T1_input_size*5, self.S1T1_input_size, rectification='ELU')
        
        self.s2 = Feedforward(self.S2T2_input_size, self.S2T2_input_size*5, self.S2T2_input_size, rectification='ELU', output_act=True)
        self.t2 = Feedforward(self.S2T2_input_size, self.S2T2_input_size*5, self.S2T2_input_size, rectification='ELU')
        
        self.log_det = 0

    def forward(self, U):
        self.log_det = 0
        D = U.shape[-1]
        D = np.int(np.floor(D/2.)) if D%2==0 else np.int(np.floor(D/2.)+1.) 
            
        log_det_1 = self.s1(U[..., D:])
        v_1 = torch.mul(U[..., :D], torch.exp(log_det_1)) + self.t1(U[..., D:])

        log_det_2 = self.s2(v_1)
        v_2 = torch.mul(U[..., D:], torch.exp(log_det_2)) + self.t2(v_1)
        V = torch.cat([v_1, v_2], axis=-1)

        self.log_det = torch.sum(torch.cat([log_det_1, log_det_2], axis=1), dim=1)

        return V

    def inverse(self, V):
        D = V.shape[-1]
        D = np.int(np.floor(D/2.)) if D%2==0 else np.int(np.floor(D/2.)+1.) 

        v_1 = V[..., :D]
        v_2 = V[..., D:]
        u_2 = torch.mul(v_2 - self.t2(v_1), torch.exp(-self.s2(v_1)))
        u_1 = torch.mul(v_1 - self.t1(u_2), torch.exp(-self.s1(u_2)))
        U = torch.cat([u_1, u_2], axis=-1)
        return U

class AffineCouplingBlock(torch.nn.Module):
    def __init__(self, data_dim, summary_dim):
        super(AffineCouplingBlock, self).__init__()

        self.S2T2_input_size = np.int(np.floor(data_dim/2.)) if data_dim%2==0 else np.int(np.floor(data_dim/2.)+1.) 
        self.S1T1_input_size = data_dim - self.S2T2_input_size        

        self.s1 = Feedforward(self.S1T1_input_size + summary_dim, (self.S1T1_input_size + summary_dim)*5, self.S1T1_input_size, rectification='ELU', output_act=True)
        self.t1 = Feedforward(self.S1T1_input_size + summary_dim, (self.S1T1_input_size + summary_dim)*5, self.S1T1_input_size, rectification='ELU')
        
        self.s2 = Feedforward(self.S2T2_input_size + summary_dim, (self.S2T2_input_size + summary_dim)*5, self.S2T2_input_size, rectification='ELU', output_act=True)
        self.t2 = Feedforward(self.S2T2_input_size + summary_dim, (self.S2T2_input_size + summary_dim)*5, self.S2T2_input_size, rectification='ELU')
        
        self.log_det = 0

    def forward(self, U):
        self.log_det = 0
        D = U[1].shape[-1]
        D = np.int(np.floor(D/2.)) if D%2==0 else np.int(np.floor(D/2.)+1.) 

        u_1 = U[1][..., :D]
        u_2 = U[1][..., D:]
        
        log_det_1 = self.s1(torch.cat([U[0], u_2], axis=1))
        v_1 = torch.mul(u_1, torch.exp(log_det_1)) + self.t1(torch.cat([U[0], u_2], axis=1))

        log_det_2 = self.s2(torch.cat([U[0], v_1], axis=1))
        v_2 = torch.mul(u_2, torch.exp(log_det_2)) + self.t2(torch.cat([U[0], v_1], axis=1))
        V = torch.cat([v_1, v_2], axis=-1)

        self.log_det = torch.sum(torch.cat([log_det_1, log_det_2], axis=1), dim=1)

        return V

    def inverse(self, V):
        D = V[1].shape[-1]
        D = np.int(np.floor(D/2.)) if D%2==0 else np.int(np.floor(D/2.)+1.) 

        v_1 = V[1][..., :D]
        v_2 = V[1][..., D:]
        u_2 = torch.mul(v_2 - self.t2(torch.cat([V[0], v_1], axis=1)), torch.exp(-self.s2(torch.cat([V[0], v_1], axis=1))))
        u_1 = torch.mul(v_1 - self.t1(torch.cat([V[0], u_2], axis=1)), torch.exp(-self.s1(torch.cat([V[0], u_2], axis=1))))
        U = torch.cat([u_2, u_1], axis=-1)
        
        return U

class INN(torch.nn.Module):
    def __init__(self, n_blocks, data_dim, batch_size):
        super(INN, self).__init__()
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.blocks = []
        self.log_dets = torch.zeros(self.batch_size) 

        for k in range(self.n_blocks):
            self.blocks.append(ACBFlow(data_dim))
            setattr(self, 'block_{}'.format(k), self.blocks[-1])

    def forward(self, x):
        y = x
        self.log_dets = torch.zeros(self.batch_size)
        for k in range(self.n_blocks):
            y = self.blocks[k](y)
            self.log_dets += self.blocks[k].log_det
        return y

    def inverse(self, y):
        x = y
        for k in range(self.n_blocks-1,-1,-1):
            x = self.blocks[k].inverse(x)
        return x

    def computeLoss(self, pred):
        loss = torch.mean(0.5*(torch.norm(pred, dim=1)**2) - self.log_dets)
        return loss

class cINN(torch.nn.Module):
    def __init__(self, net_params, n_blocks, data_dim, summary_dim, batch_size):
        super(cINN, self).__init__()
        self.summary_net = PermInvariantNetwork(*net_params)
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.blocks = []
        self.log_dets = torch.zeros(self.batch_size) 

        for k in range(self.n_blocks):
            self.blocks.append(AffineCouplingBlock(data_dim, summary_dim))
            setattr(self, 'block_{}'.format(k), self.blocks[-1])

    def forward(self, x):
        x_hat = self.summary_net(x[0])
        y = x[1]
        self.log_dets = torch.zeros(self.batch_size)
        for k in range(self.n_blocks):
            y = self.blocks[k]([x_hat, y])
            self.log_dets += self.blocks[k].log_det
        return y

    def inverse(self, y):
        y_hat = self.summary_net(y[0])
        x = y[1]
        for k in range(self.n_blocks-1,-1,-1):
            x = self.blocks[k].inverse([y_hat, x])
        return x

    def computeLoss(self, pred):
        loss = torch.mean(0.5*(torch.norm(pred, dim=1)**2) - self.log_dets)
        #loss = torch.mean((torch.norm(pred)**2)/2. - torch.sum(self.log_dets))

        return loss


