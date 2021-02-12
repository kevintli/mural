import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import imageio
from math import ceil

TARGET_POS = np.array([3., 3.], np.float32)
REPLAY_POOL_SIZE = int(5e4)

###############################
# Basic dataset visualization
###############################

def draw_env(ax=None):
    """
    Draws the medium Maze-v0 environment
    """
    ax = ax or plt.gca()
    
    ax.set_aspect('equal')
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])

    # Draw walls
    ax.add_patch(patches.Rectangle((-4/3,-4),0.2,16/3,linewidth=1,edgecolor='k',facecolor='k'))
    ax.add_patch(patches.Rectangle((4/3,-4/3),0.2,16/3,linewidth=1,edgecolor='k',facecolor='k'))
    
    ax.invert_yaxis()
    
def plot_positives(positives):
    """
    Plots a batch of positive examples as stars
    """
    plt.scatter(positives[:,0], positives[:,1], label='Goal examples', marker='*', color='g', s=200)
    
def plot_negatives(negatives, alpha=1):
    """
    Plots a batch of negative examples as 'x's
    """
    plt.scatter(negatives[:,0], negatives[:,1], label='Visited states', marker='x', color='r', s=100, alpha=alpha)
    
def plot_examples(positives, negatives, alpha=1, negative_below=True):
    if negative_below:
        plot_negatives(negatives, alpha=alpha)
    plot_positives(positives)
    if not negative_below:
        plot_negatives(negatives, alpha=alpha)
#     plt.legend(fontsize=15, bbox_to_anchor=(0.3, 0.9))
    
def discretized_states(states, bins=16, low=-4, high=4):
    """
    Converts continuous to discrete states.
    
    Params
    - states: A shape (n, 2) batch of continuous observations
    - bins: Number of bins for both x and y coordinates
    - low: Lowest value (inclusive) for continuous x and y
    - high: Highest value (inclusive) for continuous x and y
    """
    bin_size = (high - low) / bins
    shifted_states = states - low
    return np.clip(shifted_states // bin_size, 0, bins - 1).astype(np.int32)

def discrete_to_counts(states, bins=16):
    """
    Returns a shape (bins, bins) grid of visitation counts for a batch of
    discrete states.
    """
    counts = np.zeros((bins, bins))
    indices, freqs = np.unique(states, return_counts=True, axis=0)
    indices = indices.astype(np.int32)
    counts[indices[:,0], indices[:,1]] = freqs
    return counts

def plot_visitations(states, ax=None, bins=16, low=-4, high=4):
    """
    Plots a discrete grid of visitation counts given a batch of continuous x,y states.
    """
    disc = discretized_states(states, bins, low, high)
    counts = discrete_to_counts(disc, bins)
    
    ax = ax or plt.gca()
    im = ax.imshow(counts.T)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def bayesian_vice_reward(states, bins=16, low=-4, high=4):
    rewards_matrix = np.zeros((bins, bins))
    disc = discretized_states(states, bins, low, high)
    counts = discrete_to_counts(disc, bins)
    
    rewards_matrix = 1 - (counts + 1) / (counts + 2)
    dt = discretized_states(TARGET_POS, bins, low, high)
    rewards_matrix[dt[0], dt[1]] = 1

    return rewards_matrix.T

def true_bayesian_vice_reward(states, labels, bins=16, low=-4, high=4, pos_weight=1, neg_weight=1):
    rewards_matrix = np.zeros((bins, bins))
    disc = discretized_states(states, bins, low, high)
    negatives = disc[labels == 0]
    positives = disc[labels == 1]
    neg_counts = discrete_to_counts(negatives, bins)
    pos_counts = discrete_to_counts(positives, bins)
    rewards_matrix = (pos_counts + pos_weight) / (pos_counts + neg_counts + pos_weight + neg_weight)
    
    return rewards_matrix.T

def vice_reward(states, bins=16, low=-4, high=4):
    disc = discretized_states(states, bins, low, high)
    counts = discrete_to_counts(disc, bins)
    
    rewards_matrix = np.random.uniform(0, 1, counts.shape)
    rewards_matrix[counts != 0] = 1 - counts[counts != 0].astype(np.bool).astype(np.int32)
    dt = discretized_states(TARGET_POS)
    rewards_matrix[dt[0], dt[1]] = 1
    return rewards_matrix.T

def get_grid_vals(bins=16, low=-4, high=4):
    xs = np.linspace(low, high, bins)
    ys = np.linspace(low, high, bins)
    xys = np.meshgrid(xs, ys)
    grid_vals = np.array(xys).transpose(1, 2, 0).reshape(-1, 2)
    return grid_vals

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

# from swag.models.feedforward import Feedforward
# from swag.toy_dataset import ToyDataset, get_bayesian_vice_test_data
import copy
from tqdm import tqdm, trange

# toy = ToyDataset(clip_length=100, replay_pool_size=5e3)
# DEFAULT_DATA = (toy.data, toy.targets)

class Feedforward(nn.Module):
    def __init__(self, input_dim=2, num_classes=2, hidden_layers=(1024, 1024), nonlinearity=F.relu, norm=1):
        super(Feedforward, self).__init__()
        sizes = (input_dim,) + hidden_layers + (num_classes,)
        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])
        self.norm = norm or 1

    def forward(self, x):
        x /= self.norm
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.nonlinearity(x)
        x = self.layers[-1](x)
        return x

def loss_fn(recon_x, x, mu, logvar, beta=1):
    # print(torch.max(recon_x), torch.max(x))
    NLL = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)
    # NLL = torch.mean((recon_x - x) ** 2)
    # print("recon_x:", recon_x[:1])
    # print("x:", x[:1])

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = 0
    # print("NLL:", NLL)
    # print("recon_x:", recon_x)
    # print("x:", x)
    # print("KLD:", KLD)

    return NLL + KLD, NLL, KLD


def train_model_vae(model, data=None, recurring_data=None, recurring_weight=1, num_epochs=3, batch_size=128, lr=0.001,
                grad_penalty=None, beta=1, beta_schedule=None, verbose=False):
    """
    model: nn.Module
        The model to train

    data: tuple[np.array, np.array]
        A batch of training data (shape (n, d)) and labels (shape (n,)).
        If not provided, will use the ToyDataset data (from VICE replay pool).

    num_epochs: int
        Number of epochs to train for

    verbose: bool
        If True, will show loss/acc at each epoch
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda()

    data = data or DEFAULT_DATA
    # print(data[0].shape, data[1].shape)
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data[0]), torch.Tensor(data[1]).long())
    loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    nlls = []
    klds = []

    X, y = [], []

    from time import sleep

    t = trange(num_epochs, desc='Training VAE', disable=not verbose, leave=True)
    for epoch in t:
        epoch_loss = 0
        epoch_nll = 0
        epoch_kld = 0
        num_correct = 0
        num_total = 0

        curr_beta = beta
        if beta_schedule:
            for interval_end, beta_val in beta_schedule.items():
                if epoch / num_epochs <= interval_end:
                    curr_beta = beta_val
                    break

        for i, (inputs, _) in enumerate(loader):
            inputs = inputs.cuda()
            optimizer.zero_grad()
            recon_images, mu, logvar = model(inputs)
            loss, nll, kld = loss_fn(recon_images, torch.reshape(inputs, (-1, model.img_channels, 28, 28)), mu, logvar, beta=curr_beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_kld += kld.item()

        avg_loss = epoch_loss / len(loader)
        avg_nll = epoch_nll / len(loader)
        avg_kld = epoch_kld / len(loader)
        losses.append(avg_loss)
        nlls.append(avg_nll)
        klds.append(avg_kld)

        stats = {
            "Loss": '{0:.4f}'.format(avg_loss),
            "NLL": '{0:.4f}'.format(avg_nll),
            "KLD": '{0:.4f}'.format(avg_kld)
        }
        t.set_description(f"VAE | Epoch {epoch}")
        t.set_postfix(stats)
        t.refresh()
        sleep(0.01)

    # model.to(torch.device("cpu"))
    return losses, nlls, klds

def train_model(model, data=None, recurring_data=None, recurring_weight=1, num_epochs=3, batch_size=128, lr=0.001, 
                grad_penalty=None, weight_decay=0, verbose=False):
    """
    model: nn.Module
        The model to train
        
    data: tuple[np.array, np.array]
        A batch of training data (shape (n, d)) and labels (shape (n,)).
        If not provided, will use the ToyDataset data (from VICE replay pool).
        
    num_epochs: int
        Number of epochs to train for
        
    verbose: bool
        If True, will show loss/acc at each epoch
    """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    data = data or DEFAULT_DATA
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data[0]), torch.Tensor(data[1]).long())
    loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     if type(data) is torch.utils.data.dataloader.DataLoader:
#         loader = data
#     else:
#         loader = torch.utils.data.dataloader.DataLoader(torch.utils.data.TensorDataset(data[0], data[1]), batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(reduction='none' if recurring_data else 'mean')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    losses = []
    x_losses = []
    x_vals = []

    X, y = [], []

    for epoch in tqdm(range(num_epochs), disable=not verbose):
        epoch_loss = 0
        num_correct = 0
        num_total = 0

        for i, (inputs, targets) in enumerate(loader):           
            if recurring_data:
                inputs = torch.cat([inputs, torch.Tensor(recurring_data[0])])
                targets = torch.cat([targets, torch.Tensor(recurring_data[1]).long()])
            
            inputs, targets = inputs.to(device), targets.to(device)
            # X += inputs
            # y += targets
            if grad_penalty:
                inputs.requires_grad_()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if recurring_data:
                # Downweight recurring data according to total number of batches
                loss[-len(recurring_data[0]):] *= 1. * recurring_weight
                if i == 0:
                    x_losses += [torch.mean(loss[-len(recurring_data[0]):]).item()]
                    x_vals += [torch.mean(torch.softmax(outputs[-len(recurring_data[0]):], -1)[:,1]).item()]
                loss = torch.mean(loss)
                
            if grad_penalty:
                # L2 penalty on gradient of loss w.r.t. input (for smoothness)
                grad_loss = torch.autograd.grad(loss, inputs, create_graph=True)[0]
#                 print(f"Orig loss: {loss.item()}, grad penalty: {grad_penalty * torch.mean(grad_loss ** 2).item()}")
                loss += grad_penalty * torch.mean(grad_loss ** 2)
        
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_correct += len(torch.where(outputs.argmax(axis=-1) == targets)[0])
            num_total += len(outputs)
        
        print(num_correct, epoch_loss, num_total)
        acc = num_correct / num_total * 100
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        
        if verbose:
            # pass
            print(f"Epoch {epoch}: average loss {avg_loss}, accuracy {acc:.2f}%")
        
    # model.to(torch.device("cpu"))
    return (losses, (x_losses, x_vals)) if recurring_data else losses

def get_nml_probs(x, model, data=None, normalize=True, num_classes=2, query_point_weight=20, dist_weight_thresh=None, 
                  num_grad_steps=10, lr=0.01, batch_size=32, grad_penalty=None, verbose=False, 
                  show_plots=False, plotting_2d=False, return_params=False):
    """
    Returns unnormalized NML probabilities for a single input point.
    
    x: np.array
        A single input of shape (d,)
        
    model: nn.Module
        The original trained model
        
    data: tuple[np.array, np.array]
        A batch of inputs and labels (the original dataset that `model` was trained on)
        
    normalize: bool
        If True, will return normalized probabilities instead
        
    num_classes: int
        Number of classes to consider
        
    num_epochs: int
        Number of finetuning epochs for each potential class
        
    verbose: bool
        If True, will show loss/acc results at each epoch during finetuning
        
    show_plots: bool
        If True, will plot rewards (softmax probs) of original and finetuned models
    """
    results = []
    data = data or DEFAULT_DATA
    orig_inputs, orig_targets = data
    
    if show_plots and plotting_2d:
        plt.figure()
        plt.title(f"Original rewards")
        plot_rewards(model, contours=True)
        plot_dataset(data)
    
    marker_for_class = {
        0: 'x',
        1: '*'
    }
    
    model.cuda()
    num_batches = ceil(len(orig_inputs) / batch_size)

    # NOTE train on gpu, move back to cpu for eval
    
    for proposed_class in range(num_classes):
        new_model = copy.deepcopy(model)
        new_model.cuda()
        
        # Sample all of the adaptation batches in advance
        optimizer = optim.SGD(new_model.parameters(), lr=lr)
        
        for _ in range(num_grad_steps):
            idxs = np.random.permutation(range(len(orig_inputs)))[:batch_size-1]
            X, y = orig_inputs[idxs], orig_targets[idxs]
            X = torch.Tensor(np.vstack((X, x))).cuda()
            y = torch.Tensor(np.hstack((y, proposed_class))).long().cuda()
            
            logits = new_model(X)
            loss = F.cross_entropy(logits, y, reduction='none')
            
            if dist_weight_thresh:
                weights = np.exp(-np.linalg.norm(x - X.cpu().numpy(), axis=-1) * 2.3 / dist_weight_thresh)
            else:
                weights = np.ones(len(y))
                
            weights[-1] *= query_point_weight * 1. / num_batches
            weights = torch.Tensor(weights).cuda()
            loss = torch.sum(loss * weights) / torch.sum(weights)
            
            loss.backward()
            optimizer.step()
        
        new_model.cpu()
        
        with torch.no_grad():
            x_tensor = torch.Tensor(x[None])
            probs = torch.softmax(new_model(x_tensor), -1)
            results.append(probs[0][proposed_class].item())
            
        if show_plots:
            new_model.to(torch.device("cpu"))

            if plotting_2d:                
                plt.figure()
                plot_rewards(new_model, contours=True, env = False, title=f"Finetuning on label {proposed_class}")
                plot_dataset(data)
                plt.scatter(x[0], x[1], marker=marker_for_class[proposed_class], color='w', s=100)
            
            plt.figure()
            plt.title(f"Losses for label {proposed_class}")
            plt.plot(losses)
            
            plt.figure()
            plt.title(f"x loss for label {proposed_class}")
            plt.plot(x_losses)
            
            plt.figure()
            plt.title(f"x probs for label {proposed_class}")
            plt.plot(x_vals)
            
    model.cpu()
    
    if normalize:
        results = np.array(results) / sum(results)
    else:
        results = np.array(results)
    return results if not return_params else (results, new_model)

def plot_dataset(data, title=None, negative_below=True, draw_walls=True):
    if draw_walls:
        draw_env()
    inputs, targets = data
    negatives = inputs[targets == 0]
    positives = inputs[targets == 1]
    plot_examples(positives, negatives, negative_below=negative_below)
    if title:
        plt.title(title)
        
def plot_rewards(model, bins=16, low=-4, high=4, contours=False, env=True, title=None):
    """
    Plots the rewards assigned by the given model to points across the 2D environment.
    
    model: nn.Module
        A model with 2 input dimensions (x and y) and 2 output dimensions (logits)
        
    bins: int
        Number of bins for discretizing x and y values
        
    contours: bool
        If True, will plot contours instead of discretized rewards
    """
    xs = np.linspace(low, high, bins)
    ys = np.linspace(low, high, bins)
    xys = np.meshgrid(xs, ys)
    grid_vals = np.array(xys).transpose(1, 2, 0).reshape(-1, 2)

    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(grid_vals, dtype=torch.float32))

    probs = torch.softmax(logits, -1)[:,1]
    rewards = probs.reshape(bins, bins)
    
    if contours:
        if env:
            draw_env()
        else:
            plt.gca().set_aspect('equal')
            plt.gca().invert_yaxis()
        plt.contourf(xys[0], xys[1], rewards.reshape(xys[0].shape), levels=20)
    else:
        plt.imshow(rewards)
        
    plt.title(title or ("Reward contours" if contours else "Discrete rewards"))
    
    plt.clim(0, 1)
    plt.colorbar()
    
def imshow(img, low=-4, high=4, colorbar=True, clim=(0, 1), cmap='viridis'):
    plt.imshow(img, extent=[low, high, high, low], cmap=cmap)
    if colorbar:
        if clim:
            plt.clim(*clim)
        plt.colorbar(fraction=0.046, pad=0.04)
    
def contour_plot(values, low=-4, high=4, bins=16, cmap='viridis', clim=(0, 1)):
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    grid_vals = get_grid_vals(low=low, high=high, bins=bins)
    plt.contourf(grid_vals[:,0].reshape(bins, bins), grid_vals[:,1].reshape(bins, bins), values.reshape(bins, bins), cmap=cmap, levels=20, vmin=clim[0], vmax=clim[1])
    plt.colorbar()
    
def animate_plots(plot_func, num_plots, frame_duration=1., end_delay=None, fontsize=20, figsize=(8, 8), filename="plots.gif"):
    """
    plot_func : function
        Function that takes in one argument (frame index), plots any desired objects,
        then returns the frame title as a string.
        
    num_plots : int
        Number of plots (frames) to include overall
    """
    end_delay = end_delay or frame_duration
    
    imgs = []
    
    for i in range(num_plots):
#         fig = plt.figure()
        fig = plot_func(i)
#         if title:
#             fig.suptitle(title, fontsize=fontsize)
        
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) # Width, height, RGB channels
        imgs.append(data)
    
    imageio.mimsave(filename, imgs, duration=frame_duration if frame_duration == end_delay \
                    else [frame_duration] * (num_plots - 1) + [end_delay])


def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")