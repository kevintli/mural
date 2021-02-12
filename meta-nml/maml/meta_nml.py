from math import ceil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model import MetaMLPModel, VAE, MetaMNISTConvModel as MetaConvModel
from maml.metadatasets.nml import NML as NMLDataset, get_shuffled
from maml.metalearners import ModelAgnosticMetaLearning
from torchmeta.utils import gradient_update_parameters
from torchmeta.utils.data import BatchMetaDataLoader
from .kernel import KernelEmbedding
from notebook_helpers import train_model_vae

RGB_IMAGES = True

class BinaryCrossEntropy(nn.Module):
    def forward(self, pred, target, reduction='mean'):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            weights = torch.cat([(1 - target)[:,None], target[:,None]], axis=1)
        losses = torch.sum(-weights * pred, dim=-1)
        if reduction == 'mean':
            return torch.mean(losses)
        elif reduction == 'none':
            return losses
        else:
            raise Exception(f"[BCELoss] Unknown reduction type {reduction}")

bce_loss = BinaryCrossEntropy()

class MetaNML:
    def __init__(self, model=None, hidden_sizes=[1024, 1024], input_dim=2, num_classes=2,
        points_per_task=1, dist_weight_thresh=None, inner_loop_dist_weight=True, query_point_weight=1, 
        concat_query_point=False, equal_pos_neg_test=False,
        meta_lr=0.001, num_adaptation_steps=1, adaptation_step_size=0.01,
        loss_function=bce_loss, first_order=False, num_finetuning_layers=None,
        do_metalearning=True, model_vae=None, train_vae=None, embedding_type="identity",
        use_cuda=True, num_workers=8, metalearner=ModelAgnosticMetaLearning, weight_decay_lambda=0, **kwargs):
        """
        - `model` (optional), `hidden_sizes`, `input_dim`, `num_classes`:
            Model architecture to use for the classifier in MAML/NML

        - `dist_weight_thresh`, `query_point_weight`, `equal_pos_neg_test`: 
            NML training/test sampling options. See NMLDataset for details

        - `meta_lr`, `num_adaptation_steps`, `adaptation_step_size`, `loss_function`, `first_order`: 
            MAML algorithm options. See ModelAgnosticMetaLearning for details.
            Can also pass in any other desired settings as kwargs
        
        - `use_cuda`, `num_workers`: 
            Settings to use when training and evaluating NML probs.
        """
        self.concat_query_point = concat_query_point
        if self.concat_query_point:
            input_dim *= 2

        # FOR ABLATION EXPERIMENTS
        # If `self._do_metalearning` is False, this will NOT do any metatraining, and will
        # simply train an MLE model and naively take `num_adaptation_steps` gradient steps at test time.
        # All other settings, e.g. distance weighting, still apply.
        self._do_metalearning = do_metalearning
        if not self._do_metalearning:
            print("[MetaNML] Skipping metalearning!!!")

        # Initialize the model
        self.embedding_type = embedding_type
        self.train_vae = train_vae
        if self.train_vae:
            self.model = MetaConvModel(2, in_channels=3 if RGB_IMAGES else 1)
            self.model_vae = model_vae or VAE(img_channels=3 if RGB_IMAGES else 1)
        else:
            self.model = model or MetaMLPModel(input_dim, num_classes, hidden_sizes)
            self.model_vae = model_vae

        if self.model_vae:
            self.embedding_type = "vae"

        self.num_classes = num_classes
        if dist_weight_thresh:
            self.kernel = KernelEmbedding(dist_weight_thresh, self.model)
        else:
            self.kernel = None

        # Initialize MAML and the meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"[MetaNML] Using device: {self.device}")
        self.num_workers = num_workers
        self.metalearner = metalearner(
            self.model,
            self.meta_optimizer, 
            first_order=first_order, 
            num_adaptation_steps=num_adaptation_steps,
            num_finetuning_layers=num_finetuning_layers,
            step_size=adaptation_step_size,
            loss_function=loss_function,
            device=self.device,
            **kwargs
        )
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.num_adaptation_steps = num_adaptation_steps
        self.adaptation_step_size = adaptation_step_size
        self.points_per_task = points_per_task
        self.dist_weight_thresh = dist_weight_thresh
        self.inner_loop_dist_weight = inner_loop_dist_weight
        self.equal_pos_neg_test = equal_pos_neg_test
        self.query_point_weight = query_point_weight
        self._use_fixed_dataset = False
        self.weight_decay_lambda = weight_decay_lambda

    def set_fixed_dataset(self, train_data, test_data,
        task_batch_size=0, test_strategy='sample', test_batch_size=128, include_query_point=True, mixup_alpha=None):
        """
        Sets the training and test set to be fixed datasets that will be used in all future calls to train() and eval().

        Parameters
        ----------
        train_data : tuple[np.array, np.array]
            Inputs and labels to sample from for adaptation (inner loop).
            Each task will consist of one of these inputs with a proposed class label.

        test_data : tuple[np.array, np.array]
            Inputs and labels to use for the test loss (outer loop).
            The loss will be computed over ALL of these points for each task.

        (The other parameters are sampling settings to be used in the training/testing process.)
        """
        self._use_fixed_dataset = True
        self._fixed_train_data = train_data
        self._fixed_test_data = test_data
        self._fixed_task_batch_size = task_batch_size
        self._fixed_test_strategy = test_strategy
        self._fixed_test_batch_size = test_batch_size
        self._fixed_include_query_point = include_query_point
        self._fixed_mixup_alpha = mixup_alpha

    def remove_fixed_dataset(self):
        self._use_fixed_dataset = False
        self._fixed_train_data = self._fixed_test_data = None

    def train(self, train_data=None, test_data=None, batch_size=0, accumulation_steps=1, num_epochs=1, test_strategy='all', 
                test_batch_size=0, include_query_point=True, mixup_alpha=None, 
                reweight_to_uniform=False, verbose=True):
        if self._use_fixed_dataset:
            # Sample tasks from the training set
            # num_tasks = self._fixed_task_batch_size or len(self._fixed_train_data[0])
            # idxs = np.random.permutation(len(self._fixed_train_data[0]))[:num_tasks]
            tasks = self._fixed_train_data

            # Run training with the fixed dataset and settings (except for `num_epochs` and `verbose`, which can be changed)
            return self.do_train(tasks, self._fixed_test_data, batch_size=self._fixed_task_batch_size, num_epochs=num_epochs,
                test_strategy=self._fixed_test_strategy, test_batch_size=self._fixed_test_batch_size, include_query_point=self._fixed_include_query_point,
                mixup_alpha=self._fixed_mixup_alpha, verbose=verbose)
        else:
            assert (train_data is not None and test_data is not None), "Must provide train_data and test_data when not in fixed dataset mode"

            if reweight_to_uniform:
                e_Phi = np.sum(self.evaluate(train_data[0], num_grad_steps=self.num_adaptation_steps, train_data=test_data, normalize=False), axis=1)
                weights = np.clips((e_Phi - 1) / (2 - e_Phi), 0, None)
                idxs = np.random.choice(range(len(train_data[0])), len(train_data[0]), replace=True, p=weights / sum(weights))
                old_train = train_data
                train_data = (old_train[0][idxs], old_train[1][idxs])
                reweight_stats = {
                    'orig': old_train,
                    'reweighted': train_data,
                }

            results = self.do_train(train_data, test_data, batch_size=batch_size, accumulation_steps=accumulation_steps, num_epochs=num_epochs, 
                                    test_strategy=test_strategy, test_batch_size=test_batch_size, include_query_point=include_query_point, 
                                    mixup_alpha=mixup_alpha, verbose=verbose)
            if reweight_to_uniform:
                results[0] = {
                    **results[0],
                    **reweight_stats,
                }
            return results

    def train_embedding(self, train_data, num_epochs=800, verbose=True):
        if self.train_vae:
            losses, nlls, klds = train_model_vae(self.model_vae, data=train_data, num_epochs=num_epochs, verbose=verbose)
            return losses, nlls, klds
        else:
            raise Exception("[MetaNML] No embedding to train")

    def do_train(self, train_data, test_data, batch_size=0, accumulation_steps=1, num_epochs=1, test_strategy='all', 
                test_batch_size=0, include_query_point=True, mixup_alpha=None, verbose=True):
        """
        Performs `num_epochs` training iterations of meta-NML on a batch of data.

        Parameters
        ----------
        train_data : tuple[np.array, np.array]
            Inputs and labels to sample from for adaptation (inner loop).
            Each task will consist of one of these inputs with a proposed class label.

        test_data : tuple[np.array, np.array]
            Inputs and labels to use for the test loss (outer loop).
            The loss will be computed over ALL of these points for each task.

        batch_size : int
            Number of tasks to use per batch for meta-learning

        num_epochs : int
            Number of passes through the entire `train_data` dataset

        test_strategy : str in {'all', 'sample', 'cycle'}
            The strategy to use for evaluating the test loss across tasks in a batch.
            By default, we use all the points in `test_data`.

        test_batch_size : int, optional
            Number of points to use in a test batch. 
            Only used if test_strategy is 'sample' or 'cycle'.

        include_query_point : bool, optional
            Whether to include the downweighted query point in every batch during testing. Only used
            if the test_strategy is 'sample' or 'cycle'. Default: True

        mixup_alpha : float, optional
            Alpha parameter to use for mixup (https://arxiv.org/pdf/1710.09412.pdf)
            (only affects the test set, not the tasks themselves).

        verbose : bool
            Whether to show the MAML training progress bar for each epoch. Default: True
        """
        batch_size = batch_size or len(train_data[0])
        self.model.train()

        if not self._do_metalearning:
            # Do standard MLE training on the meta-test set
            ds = TensorDataset(torch.Tensor(test_data[0]), torch.Tensor(test_data[1]).long())
            loader = DataLoader(ds, batch_size=64, shuffle=True)
            epoch_results = []
            for _ in range(num_epochs):
                all_losses = []
                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    logits = self.model(inputs)
                    loss = F.cross_entropy(logits, labels)
                    self.meta_optimizer.zero_grad()
                    loss.backward()
                    self.meta_optimizer.step()
                    all_losses.append(loss.item())
                epoch_results.append({
                    'mean_loss': np.array(all_losses).mean(),
                    'all_losses': all_losses
                })
            return epoch_results    

        epoch_results = []
        for _ in range(num_epochs):
            if self.embedding_type == 'features':
                train_features = self.model.embedding(torch.Tensor(train_data[0]).cuda()).cpu().detach()
                test_features = self.model.embedding(torch.Tensor(test_data[0]).cuda()).cpu().detach()
            elif self.embedding_type == 'vae':
                train_features = self.model_vae(torch.Tensor(train_data[0]).cuda())[1].cpu().detach()
                test_features = self.model_vae(torch.Tensor(test_data[0]).cuda())[1].cpu().detach()
            elif self.embedding_type == 'custom':
                train_features = train_data[2]
                test_features = test_data[2]
            else:
                train_features = train_data[0]
                test_features = test_data[0]
            train_data = (train_data[0], train_data[1], train_features)
            test_data = (test_data[0], test_data[1], test_features)
            dataset = NMLDataset(train_data, test_data, mixup_alpha=mixup_alpha,
                                 points_per_task=self.points_per_task, num_classes=self.num_classes,
                                 test_strategy=test_strategy,
                                 test_batch_size=test_batch_size, include_query_point=include_query_point,
                                 dist_weight_thresh=self.dist_weight_thresh, equal_pos_neg_test=self.equal_pos_neg_test,
                                 query_point_weight=self.query_point_weight, kernel=self.kernel)
            trainloader = BatchMetaDataLoader(dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=self.num_workers, pin_memory=True)
            num_batches = ceil(dataset.num_tasks / batch_size)
            results = self.metalearner.train(trainloader, accumulation_steps=accumulation_steps, max_batches=num_batches, is_classification_task=True,
                verbose=verbose, desc='Training', leave=False)
            epoch_results.append(results)

        return epoch_results

    def evaluate(self, inputs, normalize=True, num_grad_steps=None, step_size=None, train_data=None, 
        include_query_point=True, cycle=False, verbose=False):
        """
        Evaluates and returns the NML probs on a batch of inputs.

        Parameters
        ----------
        inputs : np.array
            A batch of inputs to evaluate

        normalize : bool
            If True, returns the normalized output probabilities
        
        num_grad_steps : int, optional
            Number of adaptation steps to take. By default, will use the same number that MAML
            was trained with. WARNING: it tends to do worse if you take any more or less than that

        step_size : float, optional
            Step size to use. By default, will use whatever MAML was trained with.
            WARNING: it tends to do worse if you use a different value
        """
        if self._use_fixed_dataset:
            # Use the fixed test dataset to sample additional points for the adaptation batch.
            train_data = self._fixed_test_data
            include_query_point = self._fixed_include_query_point

        self.model.eval()
        results = []
        if self.embedding_type == 'custom':
            inputs = zip(inputs[0], inputs[1])
        for x in tqdm(inputs, disable=not verbose):
            results.append(self._get_metanml_probs(x, normalize=normalize, num_grad_steps=num_grad_steps, step_size=step_size, 
                train_data=train_data, include_query_point=include_query_point, cycle=cycle))
        return np.array(results)        

    def _get_metanml_probs(self, x, normalize=True, num_grad_steps=None, step_size=None, 
        train_data=None, include_query_point=True, cycle=False):
        """
        Helper function that computes the NML probs for a single input x.
        """
        if self._use_fixed_dataset:
            train_data = self._fixed_test_data

        results = []
        for proposed_class in range(self.num_classes):
            result = self._adapt(x, proposed_class, num_grad_steps=num_grad_steps, step_size=step_size, 
                train_data=train_data, include_query_point=include_query_point, cycle=cycle)
            results.append(result)
        return np.array(results) / sum(results) if normalize else np.array(results)

    def _concat_query_point(self, X, query_point):
        return np.hstack((X, np.repeat(query_point[None], len(X), axis=0)))

    def _adapt(self, x, proposed_class, num_grad_steps=None, step_size=None, return_params=False, 
        train_data=None, include_query_point=True, cycle=False):

        if self._use_fixed_dataset and train_data is None:
            train_data = self._fixed_test_data

        # if cycle:
        #     # Shuffle training data to avoid weird optimization issues
        #     train_data = get_shuffled(*train_data)

        params = None
        batch_losses = []
        batch_after_losses = []
        query_after_losses = []
        query_losses = []
        query_probs = []
        query_accs = []

        if self.embedding_type == 'custom':
            # Assumes that `x` is a tuple whose second element is the input embedding,
            # and the third element of `train_data` is a numpy array of embeddings for every training point.
            x, query_feat = x[0], torch.Tensor(x[1])[None]

        for i in range(num_grad_steps or self.num_adaptation_steps):
            input, label = torch.Tensor(x)[None], torch.Tensor([proposed_class]).long()

            if cycle:
                total_batches = ceil(len(train_data[0]) / (self.points_per_task - 1)) if include_query_point \
                    else ceil(len(train_data[0]) / (self.points_per_task))
                if i % total_batches == 0:
                    train_data = get_shuffled(*train_data)

            if self.points_per_task > 1 and train_data is not None:
                if include_query_point:
                    # Sample additional points to include in the batch
                    if cycle:
                        b = self.points_per_task - 1
                        total_batches = ceil(len(train_data[0]) / b)
                        idxs = np.arange(len(train_data[0]))[b * (i % total_batches) : b * ((i % total_batches) + 1)]
                    else:
                        idxs = np.random.permutation(range(len(train_data[0])))[:self.points_per_task - 1]
                    adapt_inputs = torch.cat([input, torch.Tensor(train_data[0][idxs])], axis=0)
                    adapt_labels = torch.cat([label, torch.Tensor(train_data[1][idxs]).long()])
                else:
                    # Treat the query point as just another point in the dataset,
                    # which may or may not be sampled for each batch
                    augmented_train_X = np.vstack([x, train_data[0]])
                    augmented_train_y = np.hstack([proposed_class, train_data[1]])
                    if cycle:
                        b = self.points_per_task
                        total_batches = ceil(len(train_data[0]) / b)
                        idxs = np.arange(len(train_data[0]))[b * (i % total_batches) : b * ((i % total_batches) + 1)]
                    else:
                        idxs = np.random.permutation(range(len(augmented_train_X)))[:self.points_per_task]
                    adapt_inputs = torch.Tensor(augmented_train_X[idxs])
                    adapt_labels = torch.Tensor(augmented_train_y[idxs]).long()

                # Compute features (for distance weighting only)
                if self.embedding_type == 'features':
                    adapt_feats = torch.Tensor(self.model.embedding(adapt_inputs.cuda()).cpu().detach())
                    query_feat = self.model.embedding(input.cuda()).cpu().detach()
                elif self.embedding_type == 'vae':
                    adapt_feats = torch.Tensor(self.model_vae(adapt_inputs.cuda())[1].cpu().detach())
                    query_feat = self.model_vae(input.cuda())[1].cpu().detach()
                elif self.embedding_type == 'custom':
                    # query_feat has already been taken from the input x
                    if include_query_point:
                        adapt_feats = torch.cat([query_feat, torch.Tensor(train_data[2][idxs])], axis=0)
                    else:
                        augmented_embeddings = np.vstack([query_feat.numpy(), train_data[2]])
                        adapt_feats = torch.Tensor(augmented_embeddings[idxs])
                else:
                    adapt_feats = adapt_inputs
                    query_feat = input

                if self.dist_weight_thresh and self.inner_loop_dist_weight:
                    # Weight according to distance, so that points near the query point are prioritized
                    weights = np.exp(-np.linalg.norm(query_feat - adapt_feats, axis=-1) * 2.3 / self.dist_weight_thresh)
                    # weights = self.kernel.embed(x, adapt_inputs.numpy())
                else:
                    weights = np.ones(len(adapt_inputs))

                if include_query_point:
                    # Downweight the query point since it's being included in every batch
                    num_batches = ceil(len(train_data[0]) / self.points_per_task)
                    weights[0] = weights[0] / num_batches * self.query_point_weight

                # Construct the training labels so that the 1st column contains actual labels, 2nd column contains weights for each point
                adapt_labels = torch.cat([adapt_labels[:,None].float(), torch.Tensor(weights)[:,None]], axis=1)
            else:
                adapt_inputs, adapt_labels = input, label

            if self.concat_query_point:
                adapt_inputs = torch.Tensor(self._concat_query_point(adapt_inputs.numpy(), x))
                input = torch.Tensor(self._concat_query_point(input.numpy(), x))

            if self.use_cuda:
                input, label = input.cuda(), label.cuda()
                adapt_inputs, adapt_labels = adapt_inputs.cuda(), adapt_labels.cuda()

            params, results = self.metalearner.adapt(adapt_inputs, adapt_labels,
                is_classification_task=True,
                num_adaptation_steps=1,
                step_size=step_size or self.metalearner.step_size, 
                first_order=True, start_params=params)
            
            batch_losses.append(results['inner_losses'].item())
            with torch.no_grad():
                logits = self.metalearner.model(input, params=params)
                prob = F.softmax(logits, dim=-1)[:,proposed_class].item()
            query_losses.append(F.cross_entropy(logits, label).item())
            query_accs.append(torch.argmax(logits).item() == label.item())
            query_probs.append(prob)

            # idxs = np.random.permutation(range(len(train_data[0])))[:self.points_per_task - 1]
            # X, y = np.vstack((x[None], train_data[0][idxs])), np.hstack((proposed_class, train_data[1][idxs]))
            # with torch.no_grad():
            #     X, y = torch.Tensor(X).cuda(), torch.Tensor(y).long().cuda()
            #     logits = self.metalearner.model(X, params=params)
            #     weights = torch.ones(len(logits)).cuda()
            #     weights[0] = self.query_point_weight
            #     batch_after_losses.append((weights * F.cross_entropy(logits, y)).mean().item())

        with torch.no_grad():
            final_logits = self.model(input, params=params)
        final_prob = F.softmax(final_logits, dim=-1)[:,proposed_class].cpu().item()

        info = {
            # 'params': params,
            'query_losses': query_losses,
            'query_accs': query_accs,
            'query_probs': query_probs,
            'batch_losses': batch_losses,
            # 'batch_after_losses': batch_after_losses,
        }

        return (final_prob, info) if return_params else final_prob
