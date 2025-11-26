import copy
import torch
from torch import nn


def dict_weight(dict1, weight):
    for k, v in dict1.items():
        dict1[k] = weight * v
    return dict1


def dict_add(dict1, dict2):
    for k, v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1


def FedAvg(global_encoder, local_models, client_weight):
    new_model_dict = None

    for client_idx in range(len(local_models)):
        local_dict = local_models[client_idx].encoder.state_dict()

        if new_model_dict is None:  
            new_model_dict = dict_weight(local_dict, client_weight[client_idx])
        else:
            new_model_dict = dict_add(new_model_dict, dict_weight(local_dict, client_weight[client_idx]))
    global_encoder.load_state_dict(new_model_dict)

    return global_encoder


