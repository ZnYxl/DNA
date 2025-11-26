import numpy as np
import torch
import time
import pdb
from models.Model import Model
from torch.autograd import Variable

def scoring_func(global_encoder, local_model, dataloader, padding_length, label_length, args):
    global_model = Model(global_encoder, args.dim, padding_length, label_length).to(
                args.device)
    global_model.eval()
    local_model.eval()

    guepi_list = torch.tensor([]).to(args.device)
    luepi_list = torch.tensor([]).to(args.device)
    guale_list = torch.tensor([]).to(args.device)
    luale_list = torch.tensor([]).to(args.device)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels= Variable(inputs.float()).to(args.device), Variable(labels).to(
                args.device)
            g_alpha = global_model(inputs) + 1
            g_strength = torch.sum(g_alpha, dim=-1, keepdim=True)
            g_prob = g_alpha / g_strength
            g_entropy = torch.sum(- g_prob * torch.log(g_prob), dim=-1)
            g_u_ale = torch.sum((g_alpha / g_strength) * (torch.digamma(g_strength + 1) - torch.digamma(g_alpha + 1)),
                                 dim=-1)
            g_u_epi = g_entropy - g_u_ale
            g_u_epi = g_u_epi.to(args.device)
            guepi_list = torch.cat((guepi_list, g_u_epi.mean(dim=1)))
            guale_list = torch.cat((guale_list, g_u_ale.mean(dim=1)))

            l_alpha = local_model(inputs) + 1
            l_strength = torch.sum(l_alpha, dim=-1, keepdim=True)
            l_prob = l_alpha / l_strength
            l_entropy = torch.sum(- l_prob * torch.log(l_prob), dim=-1)
            l_u_ale = torch.sum((l_alpha / l_strength) * (torch.digamma(l_strength + 1) - torch.digamma(l_alpha + 1)),
                                dim=-1)
            l_u_epi = l_entropy - l_u_ale
            l_u_epi = l_u_epi.to(args.device)
            luepi_list = torch.cat((luepi_list, l_u_epi.mean(dim=1)))
            luale_list = torch.cat((luale_list, l_u_ale.mean(dim=1)))

    return guepi_list.mean().cpu().numpy(), luepi_list.mean().cpu().numpy(), guale_list.mean().cpu().numpy(), luale_list.mean().cpu().numpy()