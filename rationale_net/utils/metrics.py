import sklearn.metrics

def collate_epoch_stat(stat_dict, epoch_details, mode, args):
    '''
        Update stat_dict with details from epoch_details and create
        log statement

        - stat_dict: a dictionary of statistics lists to update
        - epoch_details: list of statistics for a given epoch
        - mode: train, dev or test
        - args: model run configuration

        returns:
        -stat_dict: updated stat_dict with epoch details
        -log_statement: log statement sumarizing new epoch

    '''
    log_statement_details = ''
    for metric in epoch_details:
        loss = epoch_details[metric]
        stat_dict['{}_{}'.format(mode, metric)].append(loss)

        log_statement_details += ' -{}: {}'.format(metric, loss)

    log_statement = '\n {} - {}\n--'.format(
        args.objective, log_statement_details )

    return stat_dict, log_statement

def get_metrics(preds, golds, args):
    metrics = {}

    if args.objective  in ['cross_entropy', 'margin']:
        metrics['accuracy'] = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
        metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(y_true=golds,y_pred=preds)
        metrics['precision'] = sklearn.metrics.precision_score(y_true=golds, y_pred=preds, average="weighted")
        metrics['recall'] = sklearn.metrics.recall_score(y_true=golds,y_pred=preds, average="weighted")
        metrics['f1'] = sklearn.metrics.f1_score(y_true=golds,y_pred=preds, average="weighted")

        metrics['mse'] = "NA"

    elif args.objective == 'mse':
        metrics['mse'] = sklearn.metrics.mean_squared_error(y_true=golds, y_pred=preds)
        metrics['confusion_matrix'] = "NA"
        metrics['accuracy'] = "NA"
        metrics['precision'] = "NA"
        metrics['recall'] = "NA"
        metrics['f1'] = 'NA'

    return metrics





def init_metrics_dictionary(modes):
    '''
    Create dictionary with empty array for each metric in each mode
    '''
    epoch_stats = {}
    metrics = [
        'loss', 'obj_loss', 'k_selection_loss',
        'k_continuity_loss', 'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'mse']
    for metric in metrics:
        for mode in modes:
            key = "{}_{}".format(mode, metric)
            epoch_stats[key] = []

    return epoch_stats
