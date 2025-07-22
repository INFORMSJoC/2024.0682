import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import UserMediatorItemFM, UserMediatorSocialFM, UserMediatorFM

class CISGNN(nn.Module):
    def __init__(self, config, dataset):
        super(CISGNN, self).__init__()
        self.model_name = 'CISGNN'
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.ci_alpha = self.config['ci_alpha']
        self.n_social_layers = self.config['social_layer']

        self.interactionGraph = self.dataset.getInteractionGraph()
        self.socialGraph = self.dataset.getSocialGraph()
        # self.interPopGraph = self.dataset.getInterPopGraph()
        # self.socialPopGraph = self.dataset.getSocialPopGraph()

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()

        item_bucket_ids, item_bucket_dists = self.dataset.bucketize_items()
        user_bucket_ids, user_bucket_dists = self.dataset.bucketize_users()
        self.user_prior = torch.tensor(user_bucket_dists,
                                       dtype=torch.float32).unsqueeze(1).to(self.config['device'])
        self.item_prior = torch.tensor(item_bucket_dists,
                                       dtype=torch.float32).unsqueeze(1).to(self.config['device'])
        self.degree_num = len(self.user_prior)
        self.pop_num = len(self.item_prior)
        self.item_bucket_ids = torch.tensor(item_bucket_ids,
                                            dtype=torch.long).to(self.config['device'])
        self.user_bucket_ids = torch.tensor(user_bucket_ids,
                                            dtype=torch.long).to(self.config['device'])
        if self.config['dec_ui']=='item':
            self.mediator_encoder = UserMediatorItemFM(self.latent_dim, self.item_prior )
        elif self.config['dec_ui']=='user':
            self.mediator_encoder = UserMediatorSocialFM(self.latent_dim,  self.user_prior )
        elif self.config['dec_ui']=='ui':
            self.mediator_encoder = UserMediatorFM(self.latent_dim, self.item_prior, self.user_prior)


    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        R = self.interactionGraph
        S = self.socialGraph
        all_emb = torch.cat([users_emb, items_emb])
        inter_embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(R, all_emb)
            inter_embs.append(all_emb)
        inter_embs = torch.stack(inter_embs, dim=1)
        inter_light_out = torch.mean(inter_embs, dim=1)
        inter_users, items = torch.split(inter_light_out, [self.num_users, self.num_items])

        if self.n_social_layers != 0:
            social_embs = [users_emb]
            for layer in range(self.n_social_layers):
                users_emb_social = torch.sparse.mm(S, users_emb)
                social_embs.append(users_emb_social)
            social_embs = torch.stack(social_embs, dim=1)
            social_light_out = torch.mean(social_embs, dim=1)
            if self.n_layers != 0:
                users = inter_users + social_light_out
            else:
                users = social_light_out
        else:
            users = inter_users
        return users, items, None, None

    def computer_mediator(self, all_users, all_items):
        mean_item_embedding_matrix = self.get_group_embedding(all_items, self.item_bucket_ids, self.pop_num)
        mean_user_embedding_matrix = self.get_group_embedding(all_users, self.user_bucket_ids, self.degree_num)
        if self.config['dec_ui'] == 'item':
            mediator_emb = self.mediator_encoder(mean_item_embedding_matrix,  all_users, prior=self.config['prior'])
            return mediator_emb, None
        elif self.config['dec_ui'] == 'user':
            mediator_emb = self.mediator_encoder(mean_user_embedding_matrix, all_users, prior=self.config['prior'])
            return mediator_emb, None
        elif self.config['dec_ui'] == 'ui':
            mediator_emb = self.mediator_encoder(mean_item_embedding_matrix, mean_user_embedding_matrix, all_users,
                                                 prior=self.config['prior'])
            return mediator_emb, None



    def get_group_embedding(self, embs, group_indices, N):
        embs = embs.clone().detach()
        mean_embeddings = []
        for i in range(N):
            relevant_embeddings = embs[group_indices == i]
            if len(relevant_embeddings) > 0:
                relevant_embeddings = torch.mean(relevant_embeddings, dim=0)  # [B, H]
                mean_embeddings.append(relevant_embeddings)
        mean_embedding_matrix = torch.stack(mean_embeddings)
        norm_mean_embedding_matrix = mean_embedding_matrix.norm(p=2, dim=1, keepdim=True) + 1e-9
        mean_embedding_matrix = mean_embedding_matrix/ norm_mean_embedding_matrix
        return mean_embedding_matrix


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, h_hist, h_homo = self.computer()
        mediator_emb1, mediator_emb2 = self.computer_mediator(all_users, all_items)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, mediator_emb1, mediator_emb2, \
               users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg, timestamp):
        (users_emb, pos_emb, neg_emb, mediator_emb1, mediator_emb2,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_m_scores = (mediator_emb1[users] * pos_emb).sum(dim=-1)
        neg_m_scores = (mediator_emb1[users] * neg_emb).sum(dim=-1)
        pos_scores = (users_emb * pos_emb).sum(dim=-1)
        neg_scores = (users_emb * neg_emb).sum(dim=-1)
        pos_fuse_scores = pos_scores * torch.sigmoid(pos_m_scores)
        neg_fuse_scores = neg_scores * torch.sigmoid(neg_m_scores)
        m_loss = torch.mean(F.softplus(neg_m_scores - pos_m_scores))

        loss = torch.mean(F.softplus(neg_fuse_scores - pos_fuse_scores)) + self.ci_alpha * m_loss

        return loss, reg_loss

    def getUsersRating(self, users):
        all_users, all_items, h_hist, h_homo = self.computer()
        mediator_emb1, mediator_emb2 = self.computer_mediator(all_users, all_items)
        users_emb = all_users[users.long()]
        items_emb = all_items
        scores = torch.matmul(users_emb, items_emb.t())
        m_scores = torch.matmul(mediator_emb1[users], items_emb.t())
        R_CR = scores * torch.sigmoid(m_scores)
        return self.f(R_CR)

    def save_all_ratings(self):
        all_users, all_items, h_hist, h_homo = self.computer()
        users = torch.arange(self.num_users).long()
        all_scores = []
        for user in users:
            users_emb = all_users[user]
            items_emb = all_items
            scores = torch.matmul(users_emb, items_emb.t())
            all_scores.append(scores)
        all_scores = torch.cat(all_scores,dim=0)
        self.all_scores = all_scores

