from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks


import torch
import torch.nn as nn




class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader,  export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.ce2 = nn.CrossEntropyLoss(ignore_index=0, reduction = 'none')
        self.label2 = torch.LongTensor([0]*(args.bert_max_len-1)).to(self.device)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass



    def calculate_loss(self, batch, need_loss_per_example=False):

        seqs, labels = batch
        logits, _ = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        loss = self.ce(logits, labels.view(-1))
        if need_loss_per_example:
            batch_size = seqs.shape[0]
            #print(self.label2.repeat(batch_size, 1).shape)
            labels2 = torch.cat([self.label2.repeat(batch_size, 1), labels[:, -1].unsqueeze(-1)], dim=1)
            loss_per_example = self.ce2(logits.view(-1, logits.size(-1)), labels2.view(-1)).reshape(batch_size, -1).sum(dim = -1)
            #print(self.ce2(logits.view(-1, logits.size(-1)), labels2.view(-1)).reshape(batch_size, -1))
            return loss, loss_per_example
        else:
            return loss

        #logits2 = logits2.view(-1, logits2.size(-1))
        #loss_2=self.ce(logits2, labels)

        #query_seqs = batch[-2]
        #query_labels = batch[-1]

        
        #pairs = []
        #main_loss = 0
        #cl_loss=0
        # for i in range(num_tasks_selected):
        #     seqs=batch[2*i]
        #     labels=batch[2*i+1]
        #     logits_k,c_i_k = self.model(seqs)
        #     loss_k = self.ce(logits_k.view(-1, logits_k.size(-1)), labels.view(-1))
        #     main_loss = main_loss+loss_k
        #     pairs.append(c_i_k)
        #
        # return main_loss,main_loss

    def calculate_metrics(self, batch, metric_type):
        index = batch

        labels = self.args.matrix_label.repeat(index.shape[0],1).to(self.args.device)

        if metric_type == 'validate':
            seqs = self.args.eval_matrix_seq[index].squeeze().to(self.args.device)

            candidates = self.args.eval_matrix_candidate[index].squeeze().to(self.args.device)
        elif metric_type == 'test':
            seqs = self.args.test_matrix_seq[index].squeeze().to(self.args.device)
            candidates = self.args.test_matrix_candidate[index].squeeze().to(self.args.device)

        scores,c_is = self.model(seqs)
        scores = scores[:, -1, :]
        
        scores[:,0] = -999.999# pad token should not appear in the logits output

        scores = scores.gather(1, candidates)#the whole item set 

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

        return metrics


