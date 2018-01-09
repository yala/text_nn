#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib


POSS_VAL_NOT_LIST = "Flag {} has an invalid list of values: {}. Length of list must be >=1"


def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()

def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    '''
    jobs = [""]
    experiment_axies = []
    search_space = config['search_space']

    # Go through the tree of possible jobs and enumerate into a list of jobs
    for ind, flag in enumerate(search_space):
        possible_values = search_space[flag]
        if len(possible_values) > 1:
            experiment_axies.append(flag)

        children = []
        if len(possible_values) == 0 or type(possible_values) is not list:
            raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
        for value in possible_values:
            for parent_job in jobs:
                if type(value) is bool:
                    if value:
                        new_job_str = "{} --{}".format(parent_job, flag)
                    else:
                        new_job_str = parent_job
                elif type(value) is list:
                    val_list_str = " ".join([str(v) for v in value])
                    new_job_str = "{} --{} {}".format(parent_job, flag,
                                                      val_list_str)
                else:
                    new_job_str = "{} --{} {}".format(parent_job, flag, value)
                children.append(new_job_str)
        jobs = children

    return jobs, experiment_axies
