from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils
import copy


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.args = args
        args.num_items = len(self.smap)
        print("num_items:%d"%args.num_items)
        
        args.num_users = len(self.train)
        
        #self.attrset = set()
        #self.uniqueattr = set()
        #args.max_attr_length = 0
        #for k,v in self.item2attr.items():
        #    self.attrset = self.attrset|set(v)
        #    self.uniqueattr.add(str(v))
        #    args.max_attr_length = max(args.max_attr_length,len(v))
        #self.uniqueattr = list(self.uniqueattr)
        #self.item2uniqueattr = {}
        #for k,v in self.item2attr.items():
        #    self.item2uniqueattr[k] = self.uniqueattr.index(str(v))+1#important: reserved for padding space
            
        #args.num_attributes = len(self.attrset)
        #args.num_unique_attributes = len(self.uniqueattr)
        #print("max_attributes_length:%d"%args.max_attr_length)
        #print("num_attributes:%d"%args.num_attributes)
        #print("num_unique_attributes:%d"%args.num_unique_attributes)
        
        #self.item2attr[0] = [0]
        #self.item2uniqueattr[0] = 0
        

        self.max_len = args.bert_max_len

        self.mask_prob = args.bert_mask_prob

        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.train_slidewindow, slidewindowuserid2userid, self.user_count_slidewindow = self.get_train_dataset_slidewindow(args.slide_window_step)
        
        
        args.slidewindowuserid2userid = torch.ones(len(slidewindowuserid2userid)+1, dtype = torch.long)
        for k,v in slidewindowuserid2userid.items():
            args.slidewindowuserid2userid[k] = v        
        
        self.num_tasks_selected = args.num_tasks_selected
        
        self.users = sorted(self.train.keys())
        args.eval_matrix_seq = torch.ones(self.user_count+1, args.bert_max_len, dtype = torch.long)
        args.eval_matrix_candidate = torch.ones(self.user_count+1, args.num_items, dtype = torch.long)
        args.test_matrix_seq = torch.ones(self.user_count+1, args.bert_max_len, dtype = torch.long)
        args.test_matrix_candidate = torch.ones(self.user_count+1, args.num_items, dtype = torch.long)
        args.matrix_label = torch.LongTensor([1]+[0]*(args.num_items-1))
        #args.attribute_one_hot_label_matrix = torch.zeros(args.num_items+1, args.num_attributes+1, dtype = torch.float32)
        #args.unique_attr_matrix = torch.LongTensor(args.num_items+1)
        self.generate_eval_matrix(args)
        self.generate_test_matrix(args)
        #self.generate_attribute_one_hot_label(args)
        #self.generate_unique_attr_matrix(args)
        print("done!")
    
    def generate_eval_matrix(self,args):
        for index in range(len(self.users)):
            user = self.users[index] 

            seq = self.train[user]
            answer = self.val[user]

            seq = seq + [self.CLOZE_MASK_TOKEN]

            seq = seq[-args.bert_max_len:]

            padding_len = args.bert_max_len - len(seq)
            seq = [0] * padding_len + seq
            interacted = set(answer+seq)# remove items that user has interacted with as candidate items
            candidates = answer + [x for x in range(1,args.num_items+1) if x not in interacted]
            candidates = candidates + [0]*(args.num_items-len(candidates))#rank on the whole item set

            
            args.eval_matrix_seq[index] = torch.LongTensor(seq)
            args.eval_matrix_candidate[index] = torch.LongTensor(candidates)
            
    def generate_test_matrix(self,args):
        for index in range(len(self.users)):
            user = self.users[index] 

            seq = self.train[user]
            val = self.val[user]
            answer = self.test[user]

            seq = seq + val + [self.CLOZE_MASK_TOKEN]

            seq = seq[-args.bert_max_len:]

            padding_len = args.bert_max_len - len(seq)
            seq = [0] * padding_len + seq
            interacted = set(answer+val+seq)# remove items that user has interacted with as candidate items
            candidates = answer + [x for x in range(1,args.num_items+1) if x not in interacted]
            candidates = candidates + [0]*(args.num_items-len(candidates))#rank on the whole item set

            
            args.test_matrix_seq[index] = torch.LongTensor(seq)
            args.test_matrix_candidate[index] = torch.LongTensor(candidates)
    def generate_attribute_one_hot_label(self, args):
        for k,v in self.item2attr.items():
            for a in v:
                args.attribute_one_hot_label_matrix[k][a] = 1
    def generate_unique_attr_matrix(self, args):
        for k,v in self.item2uniqueattr.items():
            args.unique_attr_matrix[k] = v

    def get_train_dataset_slidewindow(self, step=10):
        real_user_count=0
        train_slidewindow={}
        #train_slidewindow_by_user = {}
        slidewindowuserid2userid = {}
        for user in range(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seq = [x[0] for x in self.train[user]]
            else:
                seq = self.train[user]
            seq_len = len(seq)
            #beg_idx = list(range(seq_len-self.args.bert_max_len, 0, -step))
            #beg_idx.append(0)
            beg_idx = list(range(seq_len-self.args.bert_max_len, -self.args.bert_max_len+1, -step))
            for i in beg_idx:

                #temp = seq[i:i + self.args.bert_max_len]
                temp = seq[max(i,0):i + self.args.bert_max_len]
                train_slidewindow[real_user_count] = temp
                
                slidewindowuserid2userid[real_user_count] = user

                #l = train_slidewindow_by_user.get(user,[])
                #l.append(temp)
                #train_slidewindow_by_user[user] = l

                real_user_count+=1
            '''
            all_documents[user] = [
                item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]
            ]
            '''
        return train_slidewindow, slidewindowuserid2userid, real_user_count
    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()

        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        print("users:%d"%len(self.train))
        print("pseudo users:%d"%len(self.train_slidewindow))
        dataset = BertTrainDataset(self.args,self.train_slidewindow,self.num_tasks_selected,self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):

        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size

        dataset = self._get_eval_dataset(mode)

        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode=='val':
            answers = self.val
            #dataset = BertEvalDataset(self.train, answers, self.max_len, self.item_count, self.CLOZE_MASK_TOKEN)
            dataset = BertEvalDataset(self.args, self.train, answers, self.max_len, self.item_count, self.mask_prob, self.CLOZE_MASK_TOKEN, self.rng)
        else:
            answers = self.test
            #dataset = BertTestDataset(self.train, self.val, answers, self.max_len, self.item_count, self.CLOZE_MASK_TOKEN)
            dataset = BertTestDataset(self.args, self.train, self.val, answers, self.max_len, self.item_count, self.mask_prob, self.CLOZE_MASK_TOKEN, self.rng)
        return dataset


class BertTrainDataset(data_utils.Dataset):

    def __init__(self, args, u2seq, num_tasks_selected, max_len, mask_prob, mask_token, num_items, rng):
        #self.item2attr = item2attr
        #self.item2uniqueattr = item2uniqueattr
        #self.max_attr_length = max_attr_length
        self.args = args
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.num_tasks_selected = num_tasks_selected
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng



    def __len__(self):
        return len(self.users)
    '''
    def get_masked_seq(self, seq):
        tokens = []
        labels = []

        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:

                    tokens.append(self.mask_token)

                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))

                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return tokens,labels
    '''
    def get_masked_seq(self, seq):

        if len(seq)<=3:
            tokens = copy.deepcopy(seq)
            labels = [0]*len(seq)
            labels[-1] = seq[-1]
            tokens[-1] = self.mask_token
            flag = False
        else:
            flag = True
        while flag:
            tokens = []
            labels = []
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:

                        tokens.append(self.mask_token)

                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))

                    else:
                        tokens.append(s)
                    flag = False
                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        #print(tokens, labels)
        return tokens,labels
    def __getitem__(self, index):

        user = self.users[index]
        seq = self._getseq(user)
        return_list=[]
        mask_len = self.max_len - len(seq)
        original_seq = [0] * mask_len + seq
        #return_list.append(torch.LongTensor(original_seq))
        #generate candidate tasks to select from
        for i in range(self.args.num_tasks):
            tokens, labels = self.get_masked_seq(seq)
            return_list.append(torch.LongTensor(tokens))
            return_list.append(torch.LongTensor(labels))
        
        mask_last, mask_last_label = self.get_masked_seq(seq)
        if mask_last[-1]!=self.mask_token:
            mask_last_label[-1] = mask_last[-1]
            mask_last[-1] = self.mask_token
            
        #mask_last = [0] * mask_len + seq
        #mask_last[-1] = self.mask_token
        #mask_last_label = [0]*(len(mask_last)-1) + [seq[-1]]
        
        return_list.append(torch.LongTensor(mask_last))
        return_list.append(torch.LongTensor(mask_last_label))
        return_list.append(self.args.slidewindowuserid2userid[index])
        return tuple(return_list)



    def _getseq(self, user):
        return self.u2seq[user]

'''
class BertEvalDataset(data_utils.Dataset):

    def __init__(self, u2seq, u2answer, max_len, num_items,mask_token):
        #self.item2attr = item2attr
        #self.item2uniqueattr = item2uniqueattr
        #self.max_attr_length = max_attr_length
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.num_items = num_items
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        return torch.LongTensor([index])
'''
class BertEvalDataset(data_utils.Dataset):

    def __init__(self, args, u2seq, u2answer, max_len, num_items, mask_prob, mask_token, rng):
        #self.item2attr = item2attr
        #self.item2uniqueattr = item2uniqueattr
        #self.max_attr_length = max_attr_length
        self.args = args
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.num_items = num_items
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.rng = rng

    def __len__(self):
        return len(self.users)
    def get_masked_seq(self, seq):
        tokens = []
        labels = []

        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:

                    tokens.append(self.mask_token)

                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))

                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return tokens,labels

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        return_list = []
        for i in range(self.args.num_tasks_selected):
            tokens, labels = self.get_masked_seq(seq)
            return_list.append(torch.LongTensor(tokens))
            return_list.append(torch.LongTensor(labels))
        return_list.append(torch.LongTensor([index]))
        return return_list
    
    def _getseq(self, user):
        return self.u2seq[user]

'''
class BertTestDataset(data_utils.Dataset):

    def __init__(self, u2seq, u2val, u2answer, max_len, num_items, mask_token):
        #self.item2attr = item2attr
        #self.item2uniqueattr = item2uniqueattr
        #self.max_attr_length = max_attr_length
        self.u2seq = u2seq
        self.u2val = u2val
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.num_items = num_items
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        return torch.LongTensor([index])
'''
class BertTestDataset(data_utils.Dataset):

    def __init__(self, args, u2seq, u2val, u2answer, max_len, num_items, mask_prob, mask_token, rng):
        #self.item2attr = item2attr
        #self.item2uniqueattr = item2uniqueattr
        #self.max_attr_length = max_attr_length
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.num_items = num_items
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.rng = rng

    def __len__(self):
        return len(self.users)
        
    def get_masked_seq(self, seq):
        tokens = []
        labels = []

        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:

                    tokens.append(self.mask_token)

                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))

                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return tokens,labels

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user) + self._getval(user)
        return_list = []
        for i in range(self.args.num_tasks_selected):
            tokens, labels = self.get_masked_seq(seq)
            return_list.append(torch.LongTensor(tokens))
            return_list.append(torch.LongTensor(labels))
        return_list.append(torch.LongTensor([index]))
        return return_list
    
    def _getseq(self, user):
        return self.u2seq[user]
        
    def _getval(self, user):
        return self.u2val[user]
