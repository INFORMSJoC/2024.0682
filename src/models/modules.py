import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame
from numpy import round
import math

class UserMediatorFM(nn.Module):
    def __init__(self, edim, item_pop_prior, user_degree_prior):
        super(UserMediatorFM, self).__init__()
        self.item_pop_prior = item_pop_prior
        self.user_degree_prior = user_degree_prior
        self.dim = edim
        # self.dropout = nn.Dropout(0.1)

    def forward(self, item_pop_embs, user_degree_embs, hu, prior):
        # confounder ，pop
        if prior:
            weighted_item_pop_embs = item_pop_embs * self.item_pop_prior
            weighted_user_degree_embs = user_degree_embs * self.user_degree_prior
        else:
            num_groups = len(item_pop_embs)
            item_prior = torch.tensor([1.0 / num_groups for x in range(num_groups)]).to('cuda').unsqueeze(dim=-1)
            weighted_item_pop_embs = item_pop_embs * item_prior
            num_groups = len(user_degree_embs)
            user_prior = torch.tensor([1.0 / num_groups for x in range(num_groups)]).to('cuda').unsqueeze(dim=-1)
            weighted_user_degree_embs = user_degree_embs * user_prior
        weighted_item_pop_embs_rep = weighted_item_pop_embs.expand(
            (len(hu), weighted_item_pop_embs.shape[0], hu.shape[1]))  # [F,D]-> [B, F, D]
        weighted_user_degree_embs_rep = weighted_user_degree_embs.expand(
            (len(hu), weighted_user_degree_embs.shape[0], hu.shape[1]))  # [F,D]-> [B, F, D]
        item_confounder_emb = torch.cat([hu.view(-1, 1, hu.shape[1]),
                                         weighted_item_pop_embs_rep,
                                         weighted_user_degree_embs_rep],
                                        dim=1)  # [B, 2F+1, D]
        sum_sqr_confounder_emb = item_confounder_emb.sum(dim=1).pow(2)  # B x D
        sqr_sum_confounder_emb = (item_confounder_emb.pow(2)).sum(dim=1)  # B x D
        mediator_emb = 0.5 * (sum_sqr_confounder_emb - sqr_sum_confounder_emb)  # B x D
        return mediator_emb


class UserMediatorMLP(nn.Module):
    def __init__(self, edim, item_pop_prior, user_degree_prior, item_piror):
        super(UserMediatorMLP, self).__init__()
        self.item_pop_prior = item_pop_prior
        self.user_degree_prior = user_degree_prior
        self.dim = edim
        self.item_piror = item_piror

        cate_num = len(item_pop_prior) + len(user_degree_prior) + 1
        self.confounder_fuse = nn.Sequential(
            nn.Linear(cate_num * edim, cate_num * edim // 2),
            nn.ReLU(),
            nn.Linear(cate_num * edim // 2, cate_num * edim // 4),
            nn.ReLU(),
            nn.Linear(cate_num * edim // 4, edim)
        )

    def forward(self, item_pop_embs, user_degree_embs, hu):
        weighted_item_pop_embs = item_pop_embs.reshape(1, -1).expand(len(hu), -1)
        weighted_user_degree_embs = user_degree_embs * self.user_degree_prior  # [F,D]* [F, 1]-> [F, D]
        if self.norm:
            weighted_user_degree_embs = weighted_user_degree_embs / weighted_user_degree_embs.norm(dim=1, keepdim=True)
        weighted_user_degree_embs = weighted_user_degree_embs.reshape(-1).expand(len(hu), -1)
        confounder_emb = torch.cat([hu, weighted_item_pop_embs, weighted_user_degree_embs], dim=1)  # [B, 2F+1, D]
        mediator_emb = self.confounder_fuse(confounder_emb).squeeze()
        return mediator_emb


class UserMediatorItemFM(nn.Module):
    def __init__(self, edim, item_pop_prior):
        super(UserMediatorItemFM, self).__init__()
        self.item_pop_prior = item_pop_prior
        self.dim = edim
        # self.bias_ = nn.Parameter(torch.zeros_like(item_pop_prior).to('cuda'))
        # # self.weight = nn.Parameter(torch.Tensor(len(self.item_pop_prior), edim).to('cuda'), requires_grad=True)
        # # self.reset_parameters(self.weight)
        # self.dropout = nn.Dropout(0.1)

    # def reset_parameters(self, weight):
    #     stdv = 1. / math.sqrt(weight.size(1))
    #     weight.data.uniform_(-stdv, stdv)

    def forward(self, item_pop_embs, hu, prior):
        B,D = item_pop_embs.shape
        if prior:
            weighted_item_pop_embs = item_pop_embs * self.item_pop_prior
        else:
            num_groups = len(item_pop_embs)
            item_prior = torch.tensor([1.0 / num_groups for x in range(num_groups)]).to('cuda').unsqueeze(dim=-1)
            # item_prior = self.weight / (self.weight.norm(p=2, dim=1, keepdim=True) + 1e-9)
            weighted_item_pop_embs = item_pop_embs * item_prior
        weighted_item_pop_embs = weighted_item_pop_embs.expand(
            (len(hu), weighted_item_pop_embs.shape[0], D))  # [F,D]-> [B, F, D]
        item_confounder_emb = torch.cat([hu.view(-1, 1, D),
                                         weighted_item_pop_embs
                                         ],
                                        dim=1)  # [B, 2F+1, D]
        sum_sqr_confounder_emb = item_confounder_emb.sum(dim=1).pow(2)  # B x D
        sqr_sum_confounder_emb = (item_confounder_emb.pow(2)).sum(dim=1)  # B x D
        mediator_emb = 0.5 * (sum_sqr_confounder_emb - sqr_sum_confounder_emb)  # B x D
        return mediator_emb


class UserMediatorSocialFM(nn.Module):
    def __init__(self, edim, user_degree_prior, ):
        super(UserMediatorSocialFM, self).__init__()
        self.user_degree_prior = user_degree_prior
        self.dim = edim
        # self.bias_ = nn.Parameter(torch.zeros_like(user_degree_prior).to('cuda'))

    def forward(self, user_degree_embs, hu, prior):
        B, D = user_degree_embs.shape
        if prior:
            weighted_user_degree_embs = user_degree_embs * self.user_degree_prior
        else:
            num_groups = len(user_degree_embs)
            user_prior = torch.tensor([1.0 / num_groups for x in range(num_groups)]).to('cuda').unsqueeze(dim=-1)
            weighted_user_degree_embs = user_degree_embs * user_prior
        weighted_user_degree_embs = weighted_user_degree_embs.expand(
            (len(hu), weighted_user_degree_embs.shape[0], D))  # [F,D]-> [B, F, D]
        item_confounder_emb = torch.cat([hu.view(-1, 1,D), weighted_user_degree_embs],
                                        dim=1)  # [B, 2F+1, D]
        sum_sqr_confounder_emb = item_confounder_emb.sum(dim=1).pow(2)  # B x D
        sqr_sum_confounder_emb = (item_confounder_emb.pow(2)).sum(dim=1)  # B x D
        mediator_emb = 0.5 * (sum_sqr_confounder_emb - sqr_sum_confounder_emb)  # B x D
        # fm_mediator_emb = self.dropout(mediator_emb).sum(dim=-1) + self.bias_
        return mediator_emb



class UserEncoder(nn.Module):
    def __init__(self, edim, droprate, masked_user, masked_item):
        super(UserEncoder, self).__init__()
        self.masked_user = masked_user
        self.masked_item = masked_item

        self.item_attn = AttnLayer(edim, droprate)
        self.user_attn = AttnLayer(edim, droprate)
        # self.act = nn.ReLU()

    def forward(self, user_embs, item_embs, user_emb, user_hist, user_nbrs):
        # Aggregate user history items
        hist_item_emb = item_embs(user_hist)
        hist_item_mask = -1e8 * (user_hist == self.masked_item).long()
        h_item = self.item_attn(user_emb, hist_item_emb, hist_item_mask)  # B x d
        # Aggregate user neighbors
        nbrs_emb = user_embs(user_nbrs)  # B x l x d
        nbrs_mask = -1e8 * (user_nbrs == self.masked_user).long()
        h_social = self.user_attn(user_emb, nbrs_emb, nbrs_mask)
        return h_item, h_social


class AttnLayer(nn.Module):
    def __init__(self, edim, droprate):
        super(AttnLayer, self).__init__()
        self.attn1 = nn.Linear(2 * edim, edim)
        self.attn2 = nn.Linear(edim, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, user, items, items_mask):
        user = user.unsqueeze(1).expand_as(items)
        h = torch.cat([items, user], dim=-1)
        h = self.dropout(self.act(self.attn1(h)))
        h = self.attn2(h) + items_mask.unsqueeze(-1).float()
        a = F.softmax(h, dim=1)
        attn_out = (a * items).sum(dim=1)
        return attn_out

