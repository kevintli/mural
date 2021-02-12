import torch
import math
import os
import time
import json
import logging
import pickle

from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning

def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    # Set up output folder, which will contain the saved model
    # and the config file for running test.py
    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating output folder `{0}`'.format(args.output_folder))

        folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        
        save_folder = os.path.abspath(folder)
        ckpt_folder = os.path.join(save_folder, 'checkpoints')
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
            logging.debug('Creating model checkpoint folder `{0}`'.format(ckpt_folder))

        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))

    # Load a pre-configured dataset, model, and loss function.
    benchmark = get_benchmark_by_name(args.dataset,
                                      args.folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      hidden_size=args.hidden_size)

    # Set up dataloaders:
    # MetaDataset (collection of Tasks) > Task (iterable Dataset of OrderedDicts) 
    #   > task[i] (OrderedDict with shuffled train/test split) > (tuples of input & target tensors)

    # Train loader yields batches of tasks for meta-training (both inner and outer loop)
    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    # Val loader has the same format, but is used only for evaluating adaptation ability
    # without taking gradient steps on the outer loss.
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    # Initializer the meta-optimizer and metalearner (MAML)
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=benchmark.loss_function,
                                            device=device)

    best_value = None

    def append_results_to_dict(d, results):
        for key, val in results.items():
            d[key] = d.get(key, []) + [val]

    # Training loop: each epoch goes through all tasks in the entire dataset, in batches
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    train_stats, val_stats = {}, {}
    for epoch in range(args.num_epochs):
        train_results = metalearner.train(meta_train_dataloader,
                            max_batches=args.num_batches,
                            verbose=args.verbose,
                            desc='Training',
                            leave=False)
        append_results_to_dict(train_stats, train_results)

        if epoch % args.validate_every == 0:
            val_results = metalearner.evaluate(meta_val_dataloader,
                                max_batches=args.num_batches,
                                verbose=args.verbose,
                                desc=epoch_desc.format(epoch + 1))
            append_results_to_dict(train_stats, train_results)

        if epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(ckpt_folder, f'checkpoint-{epoch}.pt')
            with open(ckpt_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)

        # Save best model according to validation acc/loss
        if 'mean_accuracy_after' in val_results:
            if (best_value is None) or (best_value < val_results['mean_accuracy_after']):
                best_value = val_results['mean_accuracy_after']
                save_model = True
        elif (best_value is None) or (best_value > val_results['mean_outer_loss']):
            best_value = val_results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)

    # Save train and val stats as serialized dictionaries
    with open(os.path.join(save_folder, 'train_stats.pkl'), 'wb') as f:
        pickle.dump(train_stats, f)
    with open(os.path.join(save_folder, 'val_stats.pkl'), 'wb') as f:
        pickle.dump(val_stats, f)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a training batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--validate-every', type=int, default=10,
        help='How often (in epochs) to evaluate model on the validation set (default: 10).')
    parser.add_argument('--checkpoint-every', type=int, default=25,
        help='How often to save a unique model checkpoint. Useful for seeing progress over time (default: 25).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
