import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import rationale_net.utils.generic as generic
import rationale_net.utils.metrics as metrics
import tqdm
import numpy as np
import pdb
import sklearn.metrics
import rationale_net.utils.train as utils

def train_model(train_data, dev_data, model, gen, args):
    '''
    Train model and tune on dev set. If model doesn't improve dev performance within args.patience
    epochs, then halve the learning rate, restore the model to best and continue training.

    At the end of training, the function will restore the model to best dev version.

    returns epoch_stats: a dictionary of epoch level metrics for train and test
    returns model : best model from this call to train
    '''

    if args.cuda:
        model = model.cuda()
        gen = gen.cuda()

    args.lr = args.init_lr
    optimizer = utils.get_optimizer([model, gen], args)

    num_epoch_sans_improvement = 0
    epoch_stats = metrics.init_metrics_dictionary(modes=['train', 'dev'])
    step = 0



    if args.class_balance:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)
        train_loader = torch.utils.data.DataLoader(
                train_data,
                num_workers=args.num_workers,
                sampler=sampler,
                batch_size=args.batch_size)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False)


    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)


    for epoch in range(1, args.epochs + 1):

        print("-------------\nEpoch {}:\n".format(epoch))
        for mode, dataset, loader in [('Train', train_data, train_loader), ('Dev', dev_data, dev_loader)]:
            train_model = mode == 'Train'
            print('{}'.format(mode))
            key_prefix = mode.lower()
            epoch_details, step, _, _, _ = run_epoch(
                data_loader=loader,
                train_model=train_model,
                model=model,
                gen=gen,
                optimizer=optimizer,
                step=step,
                args=args)

            epoch_stats, log_statement = metrics.collate_epoch_stat(epoch_stats, epoch_details, key_prefix, args)

            # Log  performance
            print(log_statement)

            if not train_model:
                print('---- Best Dev Loss is {:.4f}'.format(
                    min(epoch_stats['dev_loss'])))

        # Save model if beats best dev
        if min(epoch_stats['dev_loss']) == epoch_stats['dev_loss'][-1]:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            # Subtract one because epoch is 1-indexed and arr is 0-indexed
            epoch_stats['best_epoch'] = epoch - 1
            torch.save(model, args.model_path)
            torch.save(gen, utils.get_gen_path(args.model_path))
        else:
            num_epoch_sans_improvement += 1

        if num_epoch_sans_improvement >= args.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            model.cpu()
            gen.cpu()
            model = torch.load(args.model_path)
            gen = torch.load(utils.get_gen_path(args.model_path))

            if args.cuda:
                model = model.cuda()
                gen   = gen.cuda()
            args.lr *= .5
            optimizer = utils.get_optimizer([model, gen], args)

    # Restore model to best dev performance
    if os.path.exists(args.model_path):
        model.cpu()
        model = torch.load(args.model_path)
        gen.cpu()
        gen = torch.load(utils.get_gen_path(args.model_path))

    return epoch_stats, model, gen


def test_model(test_data, model, gen, args):
    '''
    Run model on test data, and return loss, accuracy.
    '''
    if args.cuda:
        model = model.cuda()
        gen = gen.cuda()

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    test_stats = metrics.init_metrics_dictionary(modes=['test'])

    mode = 'Test'
    train_model = False
    key_prefix = mode.lower()
    print("-------------\nTest")
    epoch_details, _, losses, preds, golds = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model,
        gen=gen,
        optimizer=None,
        step=None,
        args=args)

    test_stats, log_statement = metrics.collate_epoch_stat(test_stats, epoch_details, 'test', args)
    test_stats['losses'] = losses
    test_stats['preds'] = preds
    test_stats['golds'] = golds

    print(log_statement)

    return test_stats

def run_epoch(data_loader, train_model, model, gen, optimizer, step, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    eval_model = not train_model
    data_iter = data_loader.__iter__()

    losses = []
    obj_losses = []
    k_selection_losses = []
    k_continuity_losses = []
    preds = []
    golds = []
    losses = []

    if train_model:
        model.train()
        gen.train()
    else:
        gen.eval()
        model.eval()

    num_batches_per_epoch = len(data_iter)
    if train_model:
        num_batches_per_epoch = min(len(data_iter), 10000)

    for _ in tqdm.tqdm(range(num_batches_per_epoch)):
        batch = data_iter.next()
        if train_model:
            step += 1
            if  step % 100 == 0 or args.debug_mode:
                args.gumbel_temprature = max( np.exp((step+1) *-1* args.gumbel_decay), .05)

        x_indx = utils.get_x_indx(batch, args, eval_model)
        text = batch['text']
        indices = batch['i']
        y = autograd.Variable(batch['y'], volatile=eval_model)

        if args.cuda:
            x_indx, y = x_indx.cuda(), y.cuda()

        if train_model:
            optimizer.zero_grad()

        if args.get_rationales:
            mask, z = gen(x_indx)
        else:
            mask = None

        logit, _ = model(x_indx, mask=mask)

        if args.use_as_classifier == False:
            logit = logit.view(-1,2)
            y = y.view(-1)

        loss = get_loss(logit, y, args)
        obj_loss = loss

        if args.get_rationales:
            selection_cost, continuity_cost = gen.loss(mask, x_indx)

            loss += args.selection_lambda* selection_cost
            loss += args.continuity_lambda* continuity_cost

        if train_model:
            loss.backward()
            optimizer.step()

        if args.get_rationales:
            k_selection_losses.append( generic.tensor_to_numpy(selection_cost))
            k_continuity_losses.append( generic.tensor_to_numpy(continuity_cost))

        obj_losses.append(generic.tensor_to_numpy(obj_loss))
        losses.append( generic.tensor_to_numpy(loss) )
        preds.extend(
            torch.max(logit.data,
                      1)[1].view(y.size()).cpu().numpy())  # Record predictions
        if args.use_as_classifier == False:
            golds.extend(batch['y'].view(-1).numpy())
        else:
            golds.extend(batch['y'].numpy())

    #pdb.set_trace()
    if args.objective  in ['cross_entropy', 'margin']:
        metric = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=golds,y_pred=preds)
    elif args.objective == 'mse':
        metric = sklearn.metrics.mean_squared_error(y_true=golds, y_pred=preds)
        confusion_matrix = "NA"

    epoch_stat = {
        'loss' : np.mean(losses),
        'obj_loss': np.mean(obj_losses),
        'metric':metric,
        'confusion_matrix': confusion_matrix 
    }

    if args.get_rationales:
        epoch_stat['k_selection_loss'] = np.mean(k_selection_losses)
        epoch_stat['k_continuity_loss'] = np.mean(k_continuity_losses)

    return epoch_stat, step, losses, preds, golds


def get_loss(logit,y, args):
    if args.objective == 'cross_entropy':
        if args.use_as_classifier == False:
            loss = F.cross_entropy(logit, y, reduce=False)
            neg_loss = torch.sum(loss * (y == 0).float()) / torch.sum(y == 0).float()
            pos_loss = torch.sum(loss * (y == 1).float()) / torch.sum(y == 1).float()
            #pdb.set_trace()
            loss = (neg_loss + pos_loss) / 2
        else:
            loss = F.cross_entropy(logit, y)
    elif args.objective == 'margin':
        loss = F.multi_margin_loss(logit, y)
    elif args.objective == 'mse':
        loss = F.mse_loss(logit, y.float())
    else:
        raise Exception(
            "Objective {} not supported!".format(args.objective))
    return loss
