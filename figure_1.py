import matplotlib.pyplot as plt
from NTK_utils import network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functorch import make_functional, make_functional_with_buffers, vmap, vjp, jvp, jacrev
import IPython
import copy


def fnet_single(params, x):  # evaluates the model at a single data point
    return fnet(params, x.unsqueeze(0)).squeeze(0)


def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2)
                         for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result


def get_weights_copy(model):
    weights_path = 'weights.pth'
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_list=None):
        [data_X, data_y] = dataset
        X_tensor, y_tensor = data_X.clone().detach(), data_y.clone().detach()
        tensors = (X_tensor, y_tensor)
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transforms:
            x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_points = 200

gamma_list = torch.linspace(-3.14, 3.14, n_points)

x_list = torch.as_tensor([torch.cos(gamma) for gamma in gamma_list])
y_list = torch.as_tensor([torch.sin(gamma) for gamma in gamma_list])

x = torch.stack([x_list, y_list], dim=1)

f_x1x2 = torch.as_tensor(
    [torch.cos(gamma_list[i]) * torch.sin(gamma_list[i]) for i in range(n_points)])

dataset = CustomDataset([x, f_x1x2])


batch_size = 1

n_batches = n_points // batch_size

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

width = 500
depth = 4

layer_widths = [2, *[width for _ in range(depth - 2)], 1]


loss_fn = nn.MSELoss()


size = len(train_loader.dataset)

colors = [
    'tab:green',
    'tab:blue',
    'tab:red',
    'tab:purple'
]


for j, width in enumerate([500, 5000, 10000, 50000]):
    print(f'width: {width}')
    for i in range(10):
        print(f'init: {i}/10')
        model = network.LinearNet(layer_widths).to(device)
        fnet, params = make_functional(model)
        optimizer = optim.SGD(model.parameters(), lr=1)
        model.train()
        for batch, (X, y) in enumerate(train_loader):

            X, y = X.to(device), y.to(device)
            if batch == 0:

                model_copy = network.LinearNet(layer_widths).to(device)
                model_copy.load_state_dict(get_weights_copy(model))
                fnet, params = make_functional(model_copy)

                x1 = X.clone().detach()
                x2 = X.clone().detach()

                x2[0][0] = 1
                x2[0][1] = 0

                NTKlist = []
                for gamma in gamma_list:

                    x1[0][0] = torch.cos(gamma)
                    x1[0][1] = torch.sin(gamma)

                    NTK = empirical_ntk_jacobian_contraction(
                        fnet_single, params, x1, x2)
                    NTKlist.append(NTK[0][0][0].item())

                plt.plot(gamma_list, NTKlist, ':',
                         label=f'w = {width}, n = {batch}' if i == 5 else None, alpha=0.5, color=colors[j])
                if i == 5:
                    plt.legend()

            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if batch == 199:

                model_copy = network.LinearNet(layer_widths).to(device)
                model_copy.load_state_dict(get_weights_copy(model))
                fnet, params = make_functional(model_copy)

                x1 = X.clone().detach()
                x2 = X.clone().detach()

                x2[0][0] = 1
                x2[0][1] = 0

                NTKlist = []
                for gamma in gamma_list:

                    x1[0][0] = torch.cos(gamma)
                    x1[0][1] = torch.sin(gamma)

                    NTK = empirical_ntk_jacobian_contraction(
                        fnet_single, params, x1, x2)
                    NTKlist.append(NTK[0][0][0].item())
                plt.plot(gamma_list, NTKlist, '-',
                         label=f'w = {width}, n = {batch+1}' if i == 5 else None, alpha=0.5, color=colors[j])

                if i == 5:
                    plt.legend()

plt.title('NTK^(4) (x_0, x) vs gamma' +
          '\n x_0 = (1, 0), x = (cos(gamma), sin(gamma))')
plt.xlabel('gamma')
plt.ylabel('NTK^(4) (x_0, x)')
plt.show()
