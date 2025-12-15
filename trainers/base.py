from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .metanetwork import MetaLearner

import json
from abc import *
from pathlib import Path


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        
        #self.model_params = {}
        #for name, param in self.model.named_parameters():
        #    self.model_params[name] = copy.deepcopy(param)
        self.global_model = copy.deepcopy(self.model)
        
        self.meta_learner = MetaLearner(self.args).to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        #self.cl_loader = cl_loader
        #self.finetune_loader = finetune_loader
        self.is_parallel = False

        self.optimizer = self._create_optimizer()#local optimizer
        
        self.optimizer_global = optim.Adam(self.global_model.parameters(), lr=args.lr_global, weight_decay=args.weight_decay)
        
        self.optimizer_meta_learner = optim.Adam(self.meta_learner.parameters(), lr=args.lr_learner, weight_decay=args.weight_decay)

        if args.enable_lr_schedule:

            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)


        self.num_epochs = args.num_epochs

        self.metric_ks = args.metric_ks

        self.best_metric = args.best_metric

        self.export_root = export_root

        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()

        self.add_extra_loggers()

        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)

        self.log_period_as_iter = args.log_period_as_iter

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass


    def train(self):
        accum_iter = 0
        self.validate_meta(0,0)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch_meta(epoch, accum_iter)
            if epoch > self.args.validateafter:
                self.validate_meta(epoch,accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch_meta(self, epoch, accum_iter):
        
        
        self.model.train()

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()

        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):

            batch_size = batch[0].size(0)
            real_batch = [x.to(self.device) for x in batch]
            user_id = real_batch[-1]
            real_batch = real_batch[:-1]
            support = real_batch[:-2]
            query = real_batch[-2:]
            # initialize parameters for meta learning
            # for name, param in self.model.named_parameters():
            #    param.data.copy_(self.model_params[name])
            self.model.load_state_dict(self.global_model.state_dict())
            ##performance_evalutation
            seqs_query = query[0]
            labels_query = query[1]
            _, target_loss_before_per_example = self.calculate_loss((seqs_query, labels_query), need_loss_per_example=True)
            #print(target_loss_before_per_example)
            self.optimizer.zero_grad()
            
            # task selection
            task_logits = []
            for task in range(self.args.num_tasks):
                task_seqs = support[2 * task]
                task_labels = support[2 * task + 1]
                task_id = (task_labels > 0).float()
                # [num_candidate_tasks, batch_size]
                # user_id2 = torch.stack([user_id, user_id], dim=0)
                # task_id2 = torch.stack([task_id, task_id], dim=0)
                # print(user_id2.shape, task_id2.shape)
                weight = self.meta_learner(user_id, task_id).squeeze(-1)
                #print(weight.shape)
                task_logits.append(weight)
            #print(task_logits)
            probs = torch.stack(task_logits, dim=-1).softmax(dim=-1)  # [batch_size, num_candidate_tasks]
            #print(probs.shape)
            #print(probs)
            selected_tasks = torch.multinomial(probs, self.args.num_tasks_selected)  # [batch_size, num_tasks_selected]
            #_, selected_tasks = torch.topk(probs, 2, dim=1)
            #print(selected_tasks)
            new_support = [[] for _ in range(2 * self.args.num_tasks_selected)]
            for b in range(batch_size):
                for j in range(self.args.num_tasks_selected):
                    selected_task_indices = selected_tasks[b][j]
                    # print(selected_task_indices)
                    selected_task_seq = support[2 * selected_task_indices][b]
                    selected_task_label = support[2 * selected_task_indices + 1][b]
                    new_support[2 * j].append(selected_task_seq)
                    new_support[2 * j + 1].append(selected_task_label)
            #print(new_support)
            new_support = [torch.stack(content, dim=0) for content in new_support]
            #print(new_support[0].shape, new_support[1].shape)
            self.model.load_state_dict(self.global_model.state_dict())
            for inner_loop in range(self.args.local_update):
                #self.optimizer.zero_grad()
                for task in range(self.args.num_tasks_selected):
                    seqs_support = new_support[2*task]
                    labels_support = new_support[2*task+1]
                    inner_loss = self.calculate_loss((seqs_support, labels_support))
                    self.optimizer.zero_grad()
                    inner_loss.backward()
                    #nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    self.optimizer.step()
            seqs_query = query[0]
            labels_query = query[1]
            self.optimizer.zero_grad()
            self.optimizer_global.zero_grad()
            outer_loss, target_loss_after_per_example = self.calculate_loss((seqs_query, labels_query), need_loss_per_example=True)
            outer_loss.backward()
            global_model_named_parameters = dict(self.global_model.named_parameters())
            for name, param in self.model.named_parameters():
                param_global = global_model_named_parameters[name]
                param_global.grad = param.grad
                #global_model_named_parameters[name] = self.model_params[name] - self.lr_global_schedule[epoch]*param.grad
            self.optimizer_global.step()
            
            ##meta_learner_optimization
            if epoch >= self.args.start_to_train_meta_epoch:
            #if True:
                #since gradient update can be easily achieved, we can set a threshold value for reward function
                reward = target_loss_before_per_example.detach() - target_loss_after_per_example.detach() -0.01
                reward = reward.repeat(self.args.num_tasks_selected, 1).transpose(0,1)
                #print(reward.shape)
                log_prob = -torch.gather(probs, 1, selected_tasks).log()
                #print(log_prob)
                rl_loss = torch.mean(-torch.gather(probs, 1, selected_tasks).log()*reward)
                #print(list(self.meta_learner.parameters())[0].grad)
                self.optimizer_meta_learner.zero_grad()
                rl_loss.backward()

                self.optimizer_meta_learner.step()
                average_meter_set.update('inner_loss', inner_loss.item())
                average_meter_set.update('outer_loss', outer_loss.item())
                average_meter_set.update('rl_loss', rl_loss.item())
                average_meter_set.update('reward', reward.mean().item())
                tqdm_dataloader.set_description(
                'Epoch {}, inner_loss {:.3f}, outer_loss {:.3f}, rl_loss {:.3f} , reward {:.3f} '.format(epoch + 1, average_meter_set['inner_loss'].avg,
                                                                 average_meter_set['outer_loss'].avg,
                                                                 average_meter_set['rl_loss'].avg,
                                                                 average_meter_set['reward'].avg,))
                #print(selected_tasks)
                
            else:

                average_meter_set.update('inner_loss', inner_loss.item())
                average_meter_set.update('outer_loss', outer_loss.item())
                tqdm_dataloader.set_description(
                'Epoch {}, inner_loss {:.3f}, outer_loss {:.3f}'.format(epoch + 1, average_meter_set['inner_loss'].avg,
                                                                 average_meter_set['outer_loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter
    def train_one_epoch(self, epoch, accum_iter):

        self.model.train()

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()

        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):

            batch_size = batch[0].size(0)
            real_batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()

            loss,cl_loss = self.calculate_loss(real_batch)
            
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            average_meter_set.update('cl_loss', cl_loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f}, cl_loss {:.3f}  '.format(epoch + 1, average_meter_set['loss'].avg,
                                                                 average_meter_set['cl_loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():

            tqdm_dataloader = tqdm(self.val_loader)

            for batch_idx, batch in enumerate(tqdm_dataloader):
                #batch = batch.to(self.device)


                metrics = self.calculate_metrics(batch,'validate')

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {

                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.log_extra_val_info(log_data)
            self.logger_service.log_val(log_data)

    def validate_meta(self, epoch, accum_iter):
        average_meter_set = AverageMeterSet()

        tqdm_dataloader = tqdm(self.test_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            #batch = batch.to(self.device)
            #print(batch[0][10],self.args.test_matrix_seq[batch[-1]][10])

            self.model.load_state_dict(self.global_model.state_dict())
            self.model.train()
                
            for inner_loop in range(self.args.local_update):
                for task in range(self.args.num_tasks_selected):
                    seqs_support = (batch[2*task]).to(self.device)
                    labels_support = (batch[2*task+1]).to(self.device)
                    inner_loss = self.calculate_loss((seqs_support, labels_support))
                    self.optimizer.zero_grad()
                    inner_loss.backward()
                    #nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    self.optimizer.step()
                
            self.model.eval()
                
            metrics = self.calculate_metrics(batch[-1],'validate')

            for k, v in metrics.items():
                average_meter_set.update(k, v)
            description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                  ['Recall@%d' % k for k in self.metric_ks[:3]]
            description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
            description = description.replace('NDCG', 'N').replace('Recall', 'R')
            description = description.format(*(average_meter_set[k].avg for k in description_metrics))
            tqdm_dataloader.set_description(description)

        log_data = {

                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
        }
        log_data.update(average_meter_set.averages())
        self.log_extra_val_info(log_data)
        self.logger_service.log_val(log_data)

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')

        self.model.load_state_dict(best_model)
        self.model.eval()
        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                #batch = batch.to(self.device)

                metrics = self.calculate_metrics(batch,'test')

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)

            print(average_metrics)
    def test_meta(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')


        self.global_model.load_state_dict(best_model)  
        average_meter_set = AverageMeterSet()

        tqdm_dataloader = tqdm(self.test_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            #batch = batch.to(self.device)
            #print(batch[0][10],self.args.test_matrix_seq[batch[-1]][10])

            self.model.load_state_dict(self.global_model.state_dict())
            self.model.train()
            
                
            for inner_loop in range(self.args.local_update):
                for task in range(self.args.num_tasks_selected):
                    seqs_support = (batch[2*task]).to(self.device)
                    labels_support = (batch[2*task+1]).to(self.device)
                    inner_loss = self.calculate_loss((seqs_support, labels_support))
                    self.optimizer.zero_grad()
                    inner_loss.backward()
                    #nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    self.optimizer.step()
            
            self.model.eval()
                
            metrics = self.calculate_metrics(batch[-1],'test')

            for k, v in metrics.items():
                average_meter_set.update(k, v)
            description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                  ['Recall@%d' % k for k in self.metric_ks[:3]]
            description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
            description = description.replace('NDCG', 'N').replace('Recall', 'R')
            description = description.format(*(average_meter_set[k].avg for k in description_metrics))
            tqdm_dataloader.set_description(description)

        average_metrics = average_meter_set.averages()
        with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
            json.dump(average_metrics, f, indent=4)

        print(average_metrics)
    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.global_model.module.state_dict() if self.is_parallel else self.global_model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer_global.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
