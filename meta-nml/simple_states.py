import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import glob
import imageio

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import seaborn as sns

from maml.model import ModelMLPToy2D, MetaMNISTConvModel as MNISTModel
from maml.meta_nml import MetaNML
from maml.metadatasets.nml import get_dataset_from_file, NoisyDuplicatesProblem, TwoRingProblem, MNISTDuplicates
from notebook_helpers import *

EPOCH_NUM = 4
data = np.load(f"meta_nml_data/data_{EPOCH_NUM}.npz")
test_data = (data['test_X'], data['test_y'])


points_per_task = 256
adaptation_step_size = 0.01
meta_train_sample_size = 128
meta_test_sample_size = 2048
meta_task_batch_size = 16
meta_test_batch_size = 2048
equal_pos_neg_test = True
dist_weight_thresh = 0.5
query_point_weight = 1
test_strategy = 'all'

# Train a meta-NML model
model = ModelMLPToy2D()
meta_nml = MetaNML(model, input_dim=2, num_classes=2, points_per_task=points_per_task,
                   equal_pos_neg_test=equal_pos_neg_test,  dist_weight_thresh=dist_weight_thresh,
                   adaptation_step_size=adaptation_step_size,
                   query_point_weight=query_point_weight)
meta_nml.set_fixed_dataset((torch.from_numpy(test_data[0]).float(), torch.from_numpy(test_data[1]).float()),
                           (torch.from_numpy(test_data[0]).float(), torch.from_numpy(test_data[1]).float()),
                           task_batch_size=16,
                           test_batch_size=128)

results = []
num_epochs = 10
for ep in range(num_epochs):
    print(ep)
    results.extend(meta_nml.train())


grid_vals = get_grid_vals()
rewards = meta_nml.evaluate(grid_vals, num_grad_steps=10, train_data=test_data)[:,1].reshape(16, 16)
laplace_rewards = true_bayesian_vice_reward(data['test_X'], data['test_y'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

plt.sca(ax1)
imshow(rewards)
plt.title("Meta-NML rewards")

plt.sca(ax2)
imshow(laplace_rewards)
plt.title("Laplace smoothing rewards")

plt.sca(ax3)
imshow(rewards - laplace_rewards, cmap='RdBu', clim=(-1, 1))
plt.title("Errors (NML - laplace)")

plt.show()