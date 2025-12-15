import torch
import torch.nn as nn


class MetaLearner(nn.Module):
    def __init__(self, args):
        super(MetaLearner, self).__init__()

        self.dim = args.bert_max_len
        self.task_embedding = nn.Linear(args.bert_max_len, self.dim)
        self.user_embedding = nn.Embedding(args.num_users + 1, self.dim)
        # self.support_loss_embedding = nn.Linear(1, self.dim)
        # self.query_loss_embedding = nn.Linear(1, self.dim)
        self.fc1 = nn.Linear(2 * self.dim, self.dim)
        self.fc2 = nn.Linear(self.dim, 1)

    def forward(self, user, task):
        user_emb = self.user_embedding(user)
        task_emb = self.task_embedding(task)
        # support_loss_emb = self.support_loss_embedding(support_loss/10.0)
        # query_loss_emb = self.query_loss_embedding(query_loss/10.0)
        feature_vector = torch.cat((user_emb, task_emb), dim=-1)
        out = torch.tanh(self.fc2(self.fc1(feature_vector)))  # output value ranging from [0, 1]
        return out
