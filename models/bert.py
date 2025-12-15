from .base import BaseModel
from .bert_modules.bert import BERT

import torch
import torch.nn as nn



class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args=args
        self.bert_hidden_units = args.bert_hidden_units
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert_hidden_units, args.num_items + 1)

        
    @classmethod
    def code(cls):
        return 'bert'
    
    def get_embedding_params(self):
        return self.bert.embedding.parameters()
    
    def get_all_params(self):
        return self.parameters()
    
    def get_rec_params(self):
        return list(self.bert.transformer_blocks.parameters()) + list(self.out.parameters())
    
    def get_head_params(self):
        return self.out.parameters()
    
    
    def forward(self, x):
        c_i = self.bert(x)[-1]
        rec_output = self.out(c_i)
        return rec_output,c_i




