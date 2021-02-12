import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from maml.utils import tensors_to_device, compute_accuracy, gradient_update_parameters

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False, weight_decay_lambda=0,
                 num_adaptation_steps=1, num_finetuning_layers=None, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.num_finetuning_layers = num_finetuning_layers
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.weight_decay_lambda = weight_decay_lambda

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch, is_classification_task=False):
        """
        Performs one full MAML training iteration (inner and outer loop)
        on a batch of tasks.

        For each task in the batch:
        - Do `self.num_adaptation_steps` gradient steps on a batch of training inputs and targets
        - Evaluate the test loss on a batch of test inputs and targets
        - Update the mean test loss across tasks

        Parameters
        ----------
        batch : dict
            A dict mapping the keys 'train' and 'test' to their respective
            batches of Tasks. Each Task contains inputs and targets.

        Returns
        -------
        float
            The average test loss across tasks in the batch

        dict
            A dict with relevant training statistics ('inner_losses', 'outer_losses', 
            'accuracies_before', 'accuracies_after') as numpy arrays
        """
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point) or is_classification_task
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)
        mean_query_acc = torch.tensor(0., device=self.device)
        mean_l2_penalty = torch.tensor(0., device=self.device)
        # failed_adaptation_Xs = torch.empty(0, device=self.device)
        # failed_adaptation_ys = torch.empty(0, device=self.device, dtype=torch.long)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                if test_targets.shape[-1] == 2 and is_classification_task:
                    # If `test_targets` has 2 columns for a classification task, then the second column 
                    # will be treated as the weights for each point when computing the loss.
                    test_targets, weights = test_targets[:,0], test_targets[:,1]
                    outer_loss = self.loss_function(test_logits, test_targets, reduction='none')
                    outer_loss = torch.sum(outer_loss * weights) / torch.sum(weights)
                    # print("Sum of weights:", torch.sum(weights))
                    # print("query point:", test_inputs[-1], test_targets[-1])
                    # print("Query point NML prob:", F.softmax(test_logits, -1)[-1,test_targets[-1]])
                elif len(test_targets.shape) == 1:
                    outer_loss = self.loss_function(test_logits, test_targets)
                else:
                    raise Exception(f"Invalid target shape: {test_targets.shape}. "
                        + "Must have either 1 or 2 columns.")

                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits, test_targets)
                 
                # Store which training points we're incorrect on after adaptation
                """
                with torch.no_grad():
                    logits = self.model(train_inputs, params=params)
                    target_labels = train_targets[:,0].long() if train_targets.shape[-1] == 2 else train_targets
                    incorrect = torch.argmax(logits, dim=-1) != target_labels
                    failed_adaptation_Xs = torch.cat([failed_adaptation_Xs, train_inputs[incorrect]], axis=0)
                    failed_adaptation_ys = torch.cat([failed_adaptation_ys, target_labels[incorrect]])

                    mean_query_acc += 1 - int(incorrect[0])
                """

                
        mean_outer_loss.div_(num_tasks)
        mean_query_acc.div_(num_tasks)
        mean_l2_penalty.div_(num_tasks)

        if self.weight_decay_lambda:
            # Pre-adaptation weight decay
            l2_penalty = self._get_l2_penalty(self.model.meta_named_parameters(), self.weight_decay_lambda)
            mean_outer_loss += l2_penalty

        results['mean_outer_loss'] = mean_outer_loss.item()
        results['mean_query_acc'] = mean_query_acc.item()
        results['mean_l2_penalty'] = mean_l2_penalty.item()
        # if is_classification_task:
        #     results['failed_adaptations'] = (failed_adaptation_Xs.cpu().numpy(), failed_adaptation_ys.cpu().numpy())

        return mean_outer_loss, results

    def _get_l2_penalty(self, params, l2_reg_lambda):
        return l2_reg_lambda * sum([torch.sum(param ** 2) for param in params.values()])

    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False, start_params=None,
              **kwargs):
        """
        Performs `num_adaptation_steps` gradient steps on the given training inputs and targets.
        Does NOT actually change self.model!!

        Returns
        -------
        OrderedDict
            The model params after taking gradient steps.
            (self.model is not modified)

        dict
            Relevant training statistics ('inner_losses' and 'accuracy_before') as numpy arrays
        """
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
        params = start_params

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits = self.model(inputs, params=params)
            if targets.shape[-1] == 2 and is_classification_task:
                # If `test_targets` has 2 columns for a classification task, then the second column 
                # will be treated as the weights for each point when computing the loss.
                targets, weights = targets[:,0], targets[:,1]
                inner_loss = self.loss_function(logits, targets, reduction='none')
                inner_loss = torch.sum(inner_loss * weights) / torch.sum(weights)
            elif len(targets.shape) == 1:
                inner_loss = self.loss_function(logits, targets)
            else:
                raise Exception(f"Invalid target shape: {test_targets.shape}. "
                    + "Must have either 1 or 2 columns.")
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0) and is_classification_task:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                step_size=step_size, params=params, 
                num_layers=self.num_finetuning_layers,
                first_order=(not self.model.training) or first_order)

        return params, results

    def train(self, dataloader, accumulation_steps=1, max_batches=500, is_classification_task=False, verbose=True, **kwargs):
        """
        Runs one epoch of meta-training on the given dataset.
        Exactly `max_batches` batches of tasks will be processed.

        Returns
        -------
        dict
            The mean test loss and accuracy (if classification) over the entire dataset

        dict
            Lists of losses and accuracies (if classification) for each individual batch
        """
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        mean_query_acc = 0
        all_outer_losses = []
        all_query_accs = []
        all_l2_penalties = []
        all_failed_Xs = []
        all_failed_ys = []

        with tqdm(total=max_batches // accumulation_steps, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, accumulation_steps=accumulation_steps, max_batches=max_batches, 
                                            is_classification_task=is_classification_task):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                mean_query_acc += (results['mean_query_acc'] - mean_query_acc) / count

                all_outer_losses.append(results['mean_outer_loss'])
                all_query_accs.append(results['mean_query_acc'])
                all_l2_penalties.append(results['mean_l2_penalty'])

                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                # postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                    # postfix['accuracy'] = '{0:.4f}'.format(
                        # np.mean(results['accuracies_after']))
                if 'failed_adaptations' in results:
                    failed_Xs, failed_ys = results['failed_adaptations']
                    all_failed_Xs.extend(failed_Xs)
                    all_failed_ys.extend(failed_ys)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss, 'mean_query_acc': mean_query_acc}
        mean_results['all_outer_losses'] = all_outer_losses
        mean_results['all_query_accs'] = all_query_accs
        mean_results['all_l2_penalties'] = all_l2_penalties
        if 'accuracies_after' in results:
            mean_results['mean_accuracy_after'] = mean_accuracy
        if 'failed_adaptations' in results:
            mean_results['failed_adaptations'] = (np.array(all_failed_Xs), np.array(all_failed_ys))

        return mean_results

    def train_iter(self, dataloader, accumulation_steps=1, max_batches=500, is_classification_task=False):
        """ 
        Runs one epoch of meta-training on the given dataset, yielding
        training statistics in batches.

        Exactly `max_batches` batches of tasks will be processed.
        """
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()

        all_results = {}
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch, is_classification_task=is_classification_task)
                outer_loss /= accumulation_steps
                outer_loss.backward()

                # Add the results from this batch
                if not all_results:
                    all_results = dict(results)
                else:
                    for key in all_results:
                        all_results[key] += results[key]

                # Take a step with the accumulated gradients, and reset optimizer & results
                if (num_batches + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    all_results = {
                        key: val / accumulation_steps for key, val in all_results.items() if not key in ('num_tasks', )
                    }
                    yield all_results
                    all_results = {}

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        """
        Returns the average validation accuracy (if classification) or loss (if regression),
        measured by adaptation to the set of tasks in `dataloader`.
        """
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['mean_accuracy_after'] = mean_accuracy

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500, is_classification_task=False):
        """
        Yields the validation loss (outer loss) in batches without performing any
        gradient steps.
        """
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch, is_classification_task=is_classification_task)
                yield results

                num_batches += 1

MAML = ModelAgnosticMetaLearning

class FOMAML(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
            step_size=step_size, learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
            loss_function=loss_function, device=device)
