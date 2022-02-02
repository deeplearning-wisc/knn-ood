import torch

from ylib.yplot import plot_lines, plot_distrib

def esn(loc, scale, epsilon, num_samples):
    U = torch.rand(num_samples)
    U = U.lt((1 + epsilon) / 2).float()
    N1 = scale * torch.randn(num_samples)
    N2 = scale * torch.randn(num_samples)
    X = (1 - U) * (1 - epsilon) * N1.abs() - U * (1 + epsilon) * N2.abs()
    return X + loc

def sample_same_mu(mean, gamma, epsilon, dim, num=1000, denormalizer=None):
    mu = esn(mean, gamma, epsilon, dim)
    mu = denormalizer(mu) if denormalizer is not None else mu
    std = mu
    ret = torch.randn(num, dim) * std + mu
    ret = torch.relu(ret)
    return ret

def sample_diff_mu(mean, gamma, epsilon, dim, num=1000, denormalizer=None):
    mu = esn(mean, gamma, epsilon, dim * num).reshape((num, dim))
    mu = denormalizer(mu) if denormalizer is not None else mu
    std = mu / 2
    ret = torch.randn(num, dim) * std + mu
    ret = torch.relu(ret)
    return ret



# num_mu = 1000
# mean = 0.
# gamma = 1
# epsilon = -0.5
# places esn(-0.5, 0.5, -0.5, 512)
# plot_distrib(mu)


