import shutil
import yaml
import torch
import torch.utils.data
import torch.nn
import torch.nn.functional
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from utils.cross_entropy_with_logits import CrossEntropyLossWithLogits
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.random_sampler_custom import RandomWeightedSamplerInfinite
from utils import budget_model, dataset_bin
from utils import data_mappings_gender as dmg
from utils import data_mappings_race as dmr

torch.set_printoptions(precision=10)


class UnbiasNet:
    def __init__(self, config_filename):
        with open(config_filename, 'r') as f:
            self.cfg = yaml.load(f)

    def inference(self):
        print('Initializing models')
        model_degradation = self._init_model_degradation()

        print('Transferring models to GPU')
        if self.cfg['MODEL']['GPU_ENABLED']:
            model_degradation = torch.nn.DataParallel(model_degradation)
            model_degradation.cuda()
        model_degradation = model_degradation.eval()

        checkpoint_file = self.cfg['TEST']['CHECKPOINT_FILE']
        print('Loading checkpoint', checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        model_degradation.load_state_dict(checkpoint['model_degradation'], strict=False)

        print('Computing validation acc after degradation')
        inference_feat_csv = open(self.cfg['TEST']['INPUT_FEATURES_CSV'], 'r')
        lines = inference_feat_csv.readlines()
        keys = lines[0].strip().split(',')
        sid_idx = next(i for i, k in enumerate(keys) if k == 'SUBJECT_ID')
        fname_idx = next(i for i, k in enumerate(keys) if k == 'FILENAME')
        feat_dims = max(int(k.split('_')[-1]) for k in keys if k.startswith('DEEPFEATURE_'))
        feat_indices = [next(i for i, k in enumerate(keys) if k == f'DEEPFEATURE_{j}') for j in range(1, feat_dims + 1)]

        output_name = self.cfg['TEST']['OUTPUT_FEATURES_CSV']
        print('Output will be streamed to', output_name)
        os.makedirs(os.path.realpath(os.path.dirname(output_name)), exist_ok=True)
        fout = open(output_name, 'w')
        output_keys = ['SUBJECT_ID', 'FILENAME'] + ['DEEPFEATURE_{}'.format(idx) for idx in range(1, 257)]
        fout.write(','.join(output_keys) + '\n')

        with torch.no_grad():
            for line in tqdm(lines[1:], ncols=0, desc='Applying Degradation Model'):
                line = line.strip().split(',')
                subid = line[sid_idx]
                fname = line[fname_idx]
                feat = np.asarray([float(line[f]) for f in feat_indices], dtype=np.float64)
                feat = np.expand_dims(feat, axis=0)
                feat = torch.from_numpy(feat)
                if self.cfg['MODEL']['GPU_ENABLED']:
                    feat = feat.cuda()

                val_output_degradation = model_degradation(feat.float())
                multipass_feat = val_output_degradation.detach().cpu().numpy()

                fout.write(','.join([subid, fname] + ['{:.6f}'.format(multipass_feat[0][d]) for d in range(256)]) + '\n')

    def train(self):
        print('Building datasets')
        if self.cfg['MODEL']['TYPE'] == 'race':
            train_fc_dataloader, train_dataloader, val_task_dataloader, num_subjects, num_classes = self._init_race_dataloaders()
            meta_column_to_index = dmr.meta_column_to_index
        elif self.cfg['MODEL']['TYPE'] == 'gender':
            train_fc_dataloader, train_dataloader, val_task_dataloader, num_subjects, num_classes = self._init_gender_dataloaders()
            meta_column_to_index = dmg.meta_column_to_index
        else:
            raise RuntimeError(f'Unrecognized MODEL TYPE "{self.cfg["MODEL"]["TYPE"]}"')

        print('Initializing models')
        model_degradation = self._init_model_degradation()
        model_task = self._init_model_task(num_subjects)
        model_budget = self._init_model_budget(num_classes)

        print('Initializing objectives')
        objective_task = self._init_objective_task(self.cfg['MODEL']['TYPE'])
        objective_budget = self._init_objective_budget(self.cfg['MODEL']['TYPE'])

        if self.cfg['MODEL']['GPU_ENABLED']:
            print('Transferring models to GPU')
            model_degradation = torch.nn.DataParallel(model_degradation).cuda()
            model_task = torch.nn.DataParallel(model_task).cuda()
            model_budget = torch.nn.DataParallel(model_budget).cuda()

            objective_task = objective_task.cuda()
            objective_budget = objective_budget.cuda()

        start_step = 0

        if self.cfg['TRAIN']['RESUME_FROM_CHECKPOINT']:
            checkpoint_file = self.cfg['TRAIN']['RESUME_CHECKPOINT_FILE']
            print('Loading checkpoint', checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            model_degradation.load_state_dict(checkpoint['model_degradation'], strict=False)
            model_task.load_state_dict(checkpoint['model_task'])
            model_budget.load_state_dict(checkpoint['model_budget'])
            start_step = checkpoint['step'] + 1
            print('Successfully loaded checkpoint, starting at step', start_step)

        train_fc_data_iter = iter(train_fc_dataloader)
        train_data_iter = iter(train_dataloader)

        print('Initializing optimizers')
        optimizer_degradation = self._init_optimizer_degradation(model_degradation)
        optimizers_budget = self._init_optimizers_budget(model_budget)

        print('Initializing LR schedulers')
        equivalent_gamma = self.cfg['TRAIN']['LR_SCH_GAMMA'] ** (1.0 / self.cfg['TRAIN']['LR_SCH_DECAY_STEP'])
        lr_scheduler_degradation = torch.optim.lr_scheduler.ExponentialLR(optimizer_degradation, equivalent_gamma)
        lr_schedulers_budget = [torch.optim.lr_scheduler.ExponentialLR(optimizer_budget, equivalent_gamma) for optimizer_budget in optimizers_budget]
        for step in range(start_step, self.cfg['TRAIN']['TRAIN_STEPS']):
            torch.cuda.empty_cache()
            print('Starting step', step)

            lr_scheduler_degradation.step(1)
            for lr_scheduler_budget in lr_schedulers_budget:
                lr_scheduler_budget.step(step)

            index_budget = step % self.cfg['MODEL']['BUDGET_NUM_MODELS']

            print('LR Degradation', lr_scheduler_degradation.get_lr())
            print('LR Budget', lr_schedulers_budget[0].get_lr())

            # Initial training of FC layers on task network
            if step == 0 and self.cfg['TRAIN']['TASK_FC_TRAIN_FIRST']:
                # disable gradients for non-fc layers
                if self.cfg['MODEL']['GPU_ENABLED']:
                    non_fc_models = model_task.module[:-1]
                else:
                    non_fc_models = model_task[:-1]

                for m in non_fc_models:
                    for p in m.parameters():
                        p.requires_grad = False
                for p in model_degradation.parameters():
                    p.requires_grad = False

                optimizer_task_only = self._init_optimizer_task(model_task, model_degradation)
                objective_task_only = self._init_objective_task_only(self.cfg['MODEL']['TYPE'])

                gamma_task_fc = self.cfg['TRAIN']['TASK_FC_LR_SCH_GAMMA'] ** \
                    (1.0 / self.cfg['TRAIN']['TASK_FC_LR_SCH_DECAY_BATCHES'])
                lr_scheduler_task_fc = GradualWarmupScheduler(
                    optimizer_task_only, self.cfg['TRAIN']['TASK_FC_WARMUP_LR_SCH_MULTIPLIER'],
                    self.cfg['TRAIN']['TASK_FC_WARMUP_BATCHES'],
                    torch.optim.lr_scheduler.ExponentialLR(optimizer_task_only, gamma_task_fc))
                num_correct, num_evaluated = 0, 0
                for batch_num in range(self.cfg['TRAIN']['TASK_FC_TRAIN_BATCHES']):

                    lr_scheduler_task_fc.step()

                    data, labels = next(train_fc_data_iter)
                    if self.cfg['MODEL']['GPU_ENABLED']:
                        data, labels = data.cuda(), [l.cuda() for l in labels]

                    output_degradation = model_degradation(data)

                    output_task = model_task(output_degradation)

                    loss_task_fc = objective_task_only(output_task, labels)

                    loss_task_fc_scaled = loss_task_fc / float(self.cfg['TRAIN']['TASK_FC_BATCHES_PER_OPT_STEP'])
                    loss_task_fc_scaled.backward()
                    if (batch_num + 1) % self.cfg['TRAIN']['TASK_FC_BATCHES_PER_OPT_STEP'] == 0:
                        optimizer_task_only.step()
                        optimizer_task_only.zero_grad()

                    if (batch_num + 1) % self.cfg['TRAIN']['PRINT_FREQ'] == 0:
                        print('Training Task FC:', batch_num + 1, loss_task_fc.item(), lr_scheduler_task_fc.get_lr())
                        train_output_degradation = model_degradation(data)
                        train_output_task = model_task(train_output_degradation)

                        train_task_predicted_class = torch.argmax(train_output_task, dim=1)
                        num_correct += torch.sum(
                            (train_task_predicted_class == labels[meta_column_to_index['SUBJECT_ID']]).int())
                        num_evaluated += data.shape[0]

                        train_acc = float(num_correct) / float(num_evaluated)
                        print('TRAINACC:', num_correct.item(), num_evaluated, train_acc)
                        num_correct, num_evaluated = 0, 0
                # saving fc model
                checkpoint_name = os.path.join(
                    self.cfg['TRAIN']['CHECKPOINT_DIR'],
                    'checkpoint_fc_trained_on_all.pth.tar')
                print('Saving checkpoint to', checkpoint_name)
                os.makedirs(os.path.realpath(os.path.dirname(checkpoint_name)), exist_ok=True)
                self.save_checkpoint(
                    checkpoint_name, model_degradation, model_task, model_budget, optimizer_degradation, optimizers_budget,
                    lr_scheduler_degradation, lr_schedulers_budget, step)

                for m in non_fc_models:
                    for p in m.parameters():
                        p.requires_grad = True
                for p in model_degradation.parameters():
                    p.requires_grad = True

            # Part 0. Train all the budget models alone if starting or on restart step
            if step == 0 or (self.cfg['TRAIN']['USE_BUDGET_RESTARTING'] and step % self.cfg['TRAIN']['BUDGET_RESTART_STEP'] == 0):
                print('Retraining Budget Models')
                del optimizers_budget
                del model_budget
                del lr_schedulers_budget
                model_budget = self._init_model_budget(num_classes)
                if self.cfg['MODEL']['GPU_ENABLED']:
                    model_budget = torch.nn.DataParallel(model_budget).cuda()
                optimizers_budget = self._init_optimizers_budget(model_budget)
                lr_schedulers_budget = [torch.optim.lr_scheduler.ExponentialLR(optimizer_budget, equivalent_gamma) for optimizer_budget in optimizers_budget]

                for p in model_degradation.parameters():
                    p.requires_grad = False

                # initialize an optimizer for the budget which contains all the parameters for all the models
                optimizer_budget_all = self._init_optimizer_budget_all(model_budget)
                optimizer_budget_all.zero_grad()

                for batch_num in range(self.cfg['TRAIN']['BUDGET_INITIAL_BATCHES']):
                    data, labels = next(train_data_iter)
                    if self.cfg['MODEL']['GPU_ENABLED']:
                        data, labels = data.cuda(), [l.cuda() for l in labels]

                    output_degradation = model_degradation(data)
                    output_budgets_all = model_budget(output_degradation)

                    loss_budget_list = [objective_budget(output_budget, labels) for output_budget in output_budgets_all]
                    loss_budget = torch.sum(torch.stack(loss_budget_list))

                    loss_budget_scaled = loss_budget / float(self.cfg['TRAIN']['BUDGET_BATCHES_PER_OPT_STEP'])
                    loss_budget_scaled.backward()
                    if (batch_num + 1) % self.cfg['TRAIN']['BUDGET_BATCHES_PER_OPT_STEP'] == 0:
                        optimizer_budget_all.step()
                        optimizer_budget_all.zero_grad()

                    if (batch_num + 1) % self.cfg['TRAIN']['PRINT_FREQ'] == 0:
                        print('Train Budget Initial:', batch_num + 1, loss_budget.item())

                for p in model_degradation.parameters():
                    p.requires_grad = True

                if step == 0:
                    checkpoint_name = os.path.join(
                        self.cfg['TRAIN']['CHECKPOINT_DIR'],
                        'checkpoint_budget_trained_on_all.pth.tar')
                    print('Saving checkpoint to', checkpoint_name)
                    os.makedirs(os.path.realpath(os.path.dirname(checkpoint_name)), exist_ok=True)
                    self.save_checkpoint(
                        checkpoint_name, model_degradation, model_task, model_budget, optimizer_degradation, optimizers_budget,
                        lr_scheduler_degradation, lr_schedulers_budget, step)

            # Part 1. Train degradation alone for fixed number of iterations
            print('Training degradation model by itself')
            optimizer_degradation.zero_grad()

            for batch_num in range(self.cfg['TRAIN']['DEGRADATION_BATCHES_ALONE']):
                data, labels = next(train_data_iter)
                # print('labels:',labels)
                if self.cfg['MODEL']['GPU_ENABLED']:
                    data, labels = data.cuda(), [l.cuda() for l in labels]

                output_degradation = model_degradation(data)
                output_budget = model_budget(output_degradation)
                output_task = model_task(output_degradation)

                loss_task = objective_task(output_task, output_budget, labels)

                loss_task_scaled = loss_task / float(self.cfg['TRAIN']['TASK_BATCHES_PER_OPT_STEP'])
                loss_task_scaled.backward()
                if (batch_num + 1) % self.cfg['TRAIN']['DEGRADATION_BATCHES_PER_OPT_STEP'] == 0:
                    optimizer_degradation.step()
                    optimizer_degradation.zero_grad()

                if (batch_num + 1) % self.cfg['TRAIN']['PRINT_FREQ'] == 0:
                    print('Train Degradation:', batch_num + 1, loss_task.item())

            print('Computing validation acc after degradation')
            with torch.no_grad():
                num_correct, num_evaluated = 0, 0
                for val_data, val_labels in val_task_dataloader:
                    if self.cfg['MODEL']['GPU_ENABLED']:
                        val_data, val_labels = val_data.cuda(), [l.cuda() for l in val_labels]
                    val_output_degradation = model_degradation(val_data)
                    val_output_task = model_task(val_output_degradation)
                    val_task_predicted_class = torch.argmax(val_output_task, dim=1)
                    num_correct += torch.sum(
                        (val_task_predicted_class == val_labels[meta_column_to_index['SUBJECT_ID']]).int())
                    num_evaluated += val_data.shape[0]

                val_acc = float(num_correct) / float(num_evaluated)
                print('Validation Accuracy:', val_acc)
            print('Computing budget acc after degradation')
            with torch.no_grad():
                indx = 0
                if self.cfg['MODEL']['GPU_ENABLED']:
                    model_list = model_budget.module.models
                else:
                    model_list = model_budget.models
                for model_budget_single in model_list:
                    num_correct, num_evaluated = 0, 0
                    for val_data, val_labels in val_task_dataloader:
                        if self.cfg['MODEL']['GPU_ENABLED']:
                            val_data, val_labels = val_data.cuda(), [l.cuda() for l in val_labels]
                        val_output_degradation = model_degradation(val_data)
                        val_output_budget = model_budget_single(val_output_degradation)
                        val_output_budget_avg = sum(val_output_budget) / float(len(val_output_budget))
                        val_budget_predicted_class = torch.argmax(val_output_budget_avg, dim=1)
                        if self.cfg['MODEL']['TYPE'] == 'race':
                            val_labels_budget = val_labels[meta_column_to_index['RACE']].long()
                        else:  # 'gender'
                            val_labels_budget = (val_labels[meta_column_to_index['PR_MALE']] < 0.5).long()
                        num_correct += torch.sum((val_budget_predicted_class == val_labels_budget).int())
                        num_evaluated += val_data.shape[0]
                    val_acc = float(num_correct) / float(num_evaluated)
                    print('Budget Accuracy ' + str(indx) + ' :', val_acc)
                    indx += 1

            if step == 0:
                checkpoint_name = os.path.join(
                    self.cfg['TRAIN']['CHECKPOINT_DIR'],
                    'checkpoint_first_degradation.pth.tar')
                print('Saving checkpoint to', checkpoint_name)
                os.makedirs(os.path.realpath(os.path.dirname(checkpoint_name)), exist_ok=True)
                self.save_checkpoint(
                    checkpoint_name, model_degradation, model_task, model_budget, optimizer_degradation, optimizers_budget,
                    lr_scheduler_degradation, lr_schedulers_budget, step)

            # Part 2. Train budget models until they are too good on the training data
            if self.cfg['TRAIN']['MONITOR_BUDGET']:
                best_task_train_acc = 0.0
                task_no_increase_count = 0
                optimizer_budget = optimizers_budget[index_budget]
                if self.cfg['MODEL']['GPU_ENABLED']:
                    model_budget_single = model_budget.module.models[index_budget]
                else:
                    model_budget_single = model_budget.models[index_budget]
                while task_no_increase_count < self.cfg['TRAIN']['BUDGET_PLATEAU_THRESH']:
                    optimizer_budget.zero_grad()
                    num_correct, num_evaluated = 0, 0
                    for batch_num in range(self.cfg['TRAIN']['BUDGET_TRAIN_BATCHES_PER_CHECK']):
                        data, labels = next(train_data_iter)
                        if self.cfg['MODEL']['GPU_ENABLED']:
                            data, labels = data.cuda(), [l.cuda() for l in labels]

                        output_degradation = model_degradation(data)
                        output_budget = model_budget_single(output_degradation)

                        loss_budget = objective_budget(output_budget, labels)

                        loss_budget_scaled = loss_budget / float(self.cfg['TRAIN']['BUDGET_BATCHES_PER_OPT_STEP'])
                        loss_budget_scaled.backward()
                        if (batch_num + 1) % self.cfg['TRAIN']['BUDGET_BATCHES_PER_OPT_STEP'] == 0:
                            optimizer_budget.step()
                            optimizer_budget.zero_grad()

                        output_budget_avg = sum(output_budget) / float(len(output_budget))
                        budget_predicted_class = torch.argmax(output_budget_avg, dim=1)
                        if self.cfg['MODEL']['TYPE'] == 'race':
                            labels_budget = labels[meta_column_to_index['RACE']].long()
                        else:  # 'gender'
                            labels_budget = (labels[meta_column_to_index['PR_MALE']] < 0.5).long()
                        num_correct += torch.sum((budget_predicted_class == labels_budget).int())
                        num_evaluated += data.shape[0]

                        if (batch_num + 1) % self.cfg['TRAIN']['PRINT_FREQ'] == 0:
                            print('Train Budget:', batch_num + 1, loss_budget.item())

                    acc = float(num_correct) / float(num_evaluated)
                    print('Budget Acc:', acc)
                    if acc >= self.cfg['TRAIN']['BUDGET_MONITOR_TRAIN_ACC_THRESH']:
                        print('Training accuracy has reached threshold')
                        break

                    if acc > best_task_train_acc:
                        task_no_increase_count = 0
                        best_task_train_acc = acc
                    else:
                        task_no_increase_count += 1

                if task_no_increase_count >= self.cfg['TRAIN']['BUDGET_PLATEAU_THRESH']:
                    print('Budget training accuracy has plateaued')

            if step == 0 or (step + 1) % self.cfg['TRAIN']['SAVE_STEP'] == 0:
                checkpoint_name = os.path.join(self.cfg['TRAIN']['CHECKPOINT_DIR'],
                                               'checkpoint_{}.pth.tar'.format(step + 1))
                print('Saving checkpoint to', checkpoint_name)
                os.makedirs(os.path.realpath(os.path.dirname(checkpoint_name)), exist_ok=True)
                self.save_checkpoint(
                    checkpoint_name, model_degradation, model_task, model_budget, optimizer_degradation, optimizers_budget,
                    lr_scheduler_degradation, lr_schedulers_budget, step)
                print('Checkpoint saved to', checkpoint_name)

                most_recent_name = os.path.join(self.cfg['TRAIN']['CHECKPOINT_DIR'], 'latest.pth.tar')
                print('Copying checkpoint to', most_recent_name)
                shutil.copy(checkpoint_name, most_recent_name)

    def save_checkpoint(self, filename, model_degradation, model_task, model_budget, optimizer_degradation, optimizers_budget,
                        lr_scheduler_degradation, lr_schedulers_budget, step):
        torch.save({
            'model_degradation': model_degradation.state_dict(),
            'model_task': model_task.state_dict(),
            'model_budget': model_budget.state_dict(),
            'optimizer_degradation': optimizer_degradation.state_dict(),
            'optimizers_budget': [optimizer_budget.state_dict() for optimizer_budget in optimizers_budget],
            'lr_scheduler_degradation': lr_scheduler_degradation.state_dict(),
            'lr_schedulers_budget': [lr_scheduler_budget.state_dict() for lr_scheduler_budget in lr_schedulers_budget],
            'step': step,
        }, filename)

    def _init_optimizer_degradation(self, model_degradation):
        if self.cfg['MODEL']['GPU_ENABLED']:
            params_degradation = model_degradation.module[1].parameters()
        else:
            params_degradation = model_degradation[1].parameters()
        return torch.optim.Adam(params_degradation, lr=self.cfg['TRAIN']['INITIAL_LR_DEGRADATION'])

    def _init_optimizer_task(self, model_task, model_degradation):
        # noinspection PyTypeChecker
        if self.cfg['MODEL']['GPU_ENABLED']:
            params_degradation = model_degradation.module[1].parameters()
        else:
            params_degradation = model_degradation[1].parameters()

        return torch.optim.Adam((
            dict(params=model_task.parameters(), lr=self.cfg['TRAIN']['TASK_FC_INITIAL_LR']),
            dict(params=params_degradation, lr=self.cfg['TRAIN']['TASK_FC_INITIAL_LR'])
        ))

    def _init_optimizers_budget(self, model_budget):
        optimizers_budget = []
        num_models = self.cfg['MODEL']['BUDGET_NUM_MODELS']
        for model_index in range(num_models):
            if self.cfg['MODEL']['GPU_ENABLED']:
                model_parameters = model_budget.module.get_model_parameters(model_index)
            else:
                model_parameters = model_budget.get_model_parameters(model_index)
            optimizers_budget.append(torch.optim.Adam(model_parameters, lr=self.cfg['TRAIN']['INITIAL_LR_BUDGET']))
        return optimizers_budget

    def _init_optimizer_budget_all(self, model_budget):
        return torch.optim.Adam(model_budget.parameters(), lr=self.cfg['TRAIN']['INITIAL_LR_BUDGET'])

    def _init_objective_task(self, data_type):
        """
        Initialize the primary objective which will be used to train the task model and the degradation function.
        :return: Callable objective used by as primary objective
        """
        if data_type == 'race':
            meta_column_to_index = dmr.meta_column_to_index
        else:  # data_type == 'gender'
            meta_column_to_index = dmg.meta_column_to_index

        class TaskObjective(torch.nn.Module):
            def __init__(self, budget_loss_scale, use_budget_cross_entropy_uniform):
                super(TaskObjective, self).__init__()
                self.budget_weight = budget_loss_scale
                self.use_budget_cross_entropy_uniform = use_budget_cross_entropy_uniform
                self.task_objective = torch.nn.CrossEntropyLoss()
                self.budget_objective = CrossEntropyLossWithLogits()
                self._subject_id_col_idx = meta_column_to_index['SUBJECT_ID']

            def forward(self, output_task, output_budget, labels):
                labels_task = labels[self._subject_id_col_idx].long()
                loss_task = self.task_objective(output_task, labels_task)

                losses_budget = []
                if self.use_budget_cross_entropy_uniform:
                    # target a uniform output of budget model
                    num_classes_budget = output_budget[0][0].shape[1]
                    target_budget = torch.full(output_budget[0][0].shape,
                                               1.0 / float(num_classes_budget)).to(output_budget[0][0])
                    for output_single_budget in output_budget:
                        for output_single_budget_model in output_single_budget:
                            losses_budget.append(self.budget_objective(output_single_budget_model, target_budget))
                else:
                    # alternatively maximize the entropy of the budget output
                    for output_single_budget in output_budget:
                        for output_single_budget_model in output_single_budget:
                            softmax_budget = torch.softmax(output_single_budget_model, dim=1)
                            losses_budget.append(torch.sum(softmax_budget * torch.log(softmax_budget), dim=1))

                # only consider loss of "best" budget model (or worst in terms of task)
                print('budget losses: ', torch.stack(losses_budget))
                loss_budget = torch.max(torch.stack(losses_budget))
                return loss_task + self.budget_weight * loss_budget

        return TaskObjective(self.cfg['TRAIN']['BUDGET_LOSS_SCALE'], self.cfg['TRAIN']['BUDGET_LOSS_XENTROPY_UNIFORM'])

    def _init_objective_task_only(self, data_type):
        if data_type == 'race':
            meta_column_to_index = dmr.meta_column_to_index
        else:  # data_type == 'gender'
            meta_column_to_index = dmg.meta_column_to_index

        class TaskObjectiveFCOnly(torch.nn.Module):
            def __init__(self):
                super(TaskObjectiveFCOnly, self).__init__()
                self.task_objective = torch.nn.CrossEntropyLoss()
                self._label_col_idx = meta_column_to_index['SUBJECT_ID']

            def forward(self, output_task, labels):
                labels_task = labels[self._label_col_idx].long()
                return self.task_objective(output_task, labels_task)

        return TaskObjectiveFCOnly()

    def _init_objective_budget(self, data_type):
        """
        Initialize the objective used by the budget networks.
        :return: Callable objective used to train the budget networks.
        """
        if data_type == 'race':
            class BudgetObjective(torch.nn.Module):
                def __init__(self):
                    super(BudgetObjective, self).__init__()
                    self.objective = torch.nn.CrossEntropyLoss()

                def forward(self, output_budget, labels):
                    labels_budget = labels[dmr.meta_column_to_index['RACE']].long()
                    loss_total = 0.0
                    for output_single_budget in output_budget:
                        loss_total = loss_total + self.objective(output_single_budget, labels_budget)
                    return loss_total
        else:  # data_type == 'gender':
            class BudgetObjective(torch.nn.Module):
                def __init__(self):
                    super(BudgetObjective, self).__init__()
                    self.objective = torch.nn.CrossEntropyLoss()

                def forward(self, output_budget, labels):
                    labels_budget = (labels[dmg.meta_column_to_index['PR_MALE']] < 0.5).long()
                    loss_total = 0.0
                    for output_single_budget in output_budget:
                        loss_total = loss_total + self.objective(output_single_budget, labels_budget)
                    return loss_total

        return BudgetObjective()

    def _init_race_datasets(self):
        train_dataset = dataset_bin.DatasetBin(self.cfg['TRAIN']['TRAIN_META'], self.cfg['TRAIN']['TRAIN_BIN_FEATS'], dmr.meta_columns)
        if self.cfg['TRAIN']['VAL_META'] == self.cfg['TRAIN']['TRAIN_META'] and self.cfg['TRAIN']['VAL_BIN_FEATS'] == self.cfg['TRAIN']['TRAIN_BIN_FEATS']:
            print('val and train dataset are the same')
            val_dataset = train_dataset
        else:
            val_dataset = dataset_bin.DatasetBin(self.cfg['TRAIN']['VAL_META'], self.cfg['TRAIN']['VAL_BIN_FEATS'], dmr.meta_columns)
        return train_dataset, val_dataset

    def _init_gender_datasets(self):
        train_dataset = dataset_bin.DatasetBin(self.cfg['TRAIN']['TRAIN_META'], self.cfg['TRAIN']['TRAIN_BIN_FEATS'], dmg.meta_columns)
        if self.cfg['TRAIN']['VAL_META'] == self.cfg['TRAIN']['TRAIN_META'] and self.cfg['TRAIN']['VAL_BIN_FEATS'] == self.cfg['TRAIN']['TRAIN_BIN_FEATS']:
            print('val and train dataset are the same')
            val_dataset = train_dataset
        else:
            val_dataset = dataset_bin.DatasetBin(self.cfg['TRAIN']['VAL_META'], self.cfg['TRAIN']['VAL_BIN_FEATS'], dmg.meta_columns)
        return train_dataset, val_dataset

    def _init_race_dataloaders(self):
        train_dataset, val_dataset = self._init_race_datasets()

        subject_id_index = dmr.meta_column_to_index['SUBJECT_ID']
        race_index = dmr.meta_column_to_index['RACE']

        train_subject_count = defaultdict(lambda: 0)
        train_subject_race_count = defaultdict(lambda: defaultdict(lambda: 0))
        train_race = list()
        for target, _ in tqdm(train_dataset.samples, ncols=0, desc='Counting subjects'):
            subject_id = target[subject_id_index]
            train_subject_count[subject_id] += 1
            train_race.append(target[race_index])
            train_subject_race_count[train_race[-1]][subject_id] += 1

        train_sampler_weights = [0] * len(train_dataset)
        for index, (target, _) in tqdm(enumerate(train_dataset.samples), total=len(train_dataset), ncols=0,
                                       desc='Computing subject weights'):
            subject_id = int(target[subject_id_index])
            num_subjects_of_race = len(train_subject_race_count[train_race[index]])
            num_instance_of_subject = train_subject_count[subject_id]
            train_sampler_weights[index] = 1.0 / (num_subjects_of_race * num_instance_of_subject)

        print('Initializing train dataloader')
        train_sampler = RandomWeightedSamplerInfinite(train_dataset, train_sampler_weights)
        train_fc_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg['TRAIN']['FC_BATCH_SIZE'], sampler=train_sampler,
            num_workers=self.cfg['TRAIN']['NUM_WORKERS'], pin_memory=bool(self.cfg['TRAIN']['PIN_MEMORY'])
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg['TRAIN']['TRAIN_BATCH_SIZE'], sampler=train_sampler,
            num_workers=self.cfg['TRAIN']['NUM_WORKERS'], pin_memory=bool(self.cfg['TRAIN']['PIN_MEMORY'])
        )

        val_subject_count = defaultdict(lambda: 0)
        val_race = list()
        val_subject_race_count = defaultdict(lambda: defaultdict(lambda: 0))
        for target, _ in tqdm(val_dataset.samples, ncols=0, desc='Counting subjects'):
            subject_id = target[subject_id_index]
            val_subject_count[subject_id] += 1
            val_race.append(target[race_index])
            val_subject_race_count[val_race[-1]][subject_id] += 1
        val_sampler_weights = [0] * len(val_dataset)
        for index, (target, _) in tqdm(enumerate(val_dataset.samples), total=len(val_dataset), ncols=0,
                                       desc='Computing subject weights'):
            subject_id = int(target[subject_id_index])
            num_subjects_of_race = len(val_subject_race_count[val_race[index]])
            num_instance_of_subject = val_subject_count[subject_id]
            val_sampler_weights[index] = 1.0 / (num_subjects_of_race * num_instance_of_subject)

        print('Initializing val dataloader')
        val_sampler = torch.utils.data.WeightedRandomSampler(
            val_sampler_weights,
            num_samples=self.cfg['TRAIN']['VAL_BATCH_SIZE'] * self.cfg['TRAIN']['TASK_VAL_PROBE_BATCHES'])
        val_task_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.cfg['TRAIN']['VAL_BATCH_SIZE'],
            sampler=val_sampler, num_workers=self.cfg['TRAIN']['NUM_WORKERS'],
            pin_memory=bool(self.cfg['TRAIN']['PIN_MEMORY'])
        )

        return train_fc_dataloader, train_dataloader, val_task_dataloader, train_dataset.num_subjects, 4

    def _init_gender_dataloaders(self):
        train_dataset, val_dataset = self._init_gender_datasets()

        # balance classes in training data
        subject_id_index = dmg.meta_column_to_index['SUBJECT_ID']
        pr_male_index = dmg.meta_column_to_index['PR_MALE']
        train_subject_count = defaultdict(lambda: 0)
        train_gender_count = defaultdict(lambda: 0)
        for target, _ in tqdm(train_dataset.samples, ncols=0, desc='Counting subjects'):
            subject_id = int(target[subject_id_index])
            is_male = float(target[pr_male_index]) >= 0.5
            train_subject_count[subject_id] += 1
            train_gender_count[is_male] += 1
        train_sampler_weights = [0] * len(train_dataset)
        for index, (target, _) in tqdm(enumerate(train_dataset.samples), total=len(train_dataset), ncols=0,
                                       desc='Computing subject weights'):
            subject_id = int(target[subject_id_index])
            is_male = float(target[pr_male_index]) >= 0.5
            train_sampler_weights[index] = 1.0 / train_subject_count[subject_id] / float(train_gender_count[is_male])

        print('Initializing train dataloader')
        train_sampler = RandomWeightedSamplerInfinite(train_dataset, train_sampler_weights)
        train_fc_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg['TRAIN']['FC_BATCH_SIZE'], sampler=train_sampler,
            num_workers=self.cfg['TRAIN']['NUM_WORKERS'], pin_memory=bool(self.cfg['TRAIN']['PIN_MEMORY'])
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg['TRAIN']['TRAIN_BATCH_SIZE'], sampler=train_sampler,
            num_workers=self.cfg['TRAIN']['NUM_WORKERS'], pin_memory=bool(self.cfg['TRAIN']['PIN_MEMORY'])
        )

        val_subject_count = defaultdict(lambda: 0)
        val_gender_count = defaultdict(lambda: 0)
        for target, _ in tqdm(val_dataset.samples, ncols=0, desc='Counting subjects'):
            subject_id = int(target[subject_id_index])
            is_male = float(target[pr_male_index]) >= 0.5
            val_subject_count[subject_id] += 1
            val_gender_count[is_male] += 1
        val_sampler_weights = [0] * len(val_dataset)
        for index, (target, _) in tqdm(enumerate(val_dataset.samples), total=len(val_dataset), ncols=0,
                                       desc='Computing subject weights'):
            subject_id = int(target[subject_id_index])
            is_male = float(target[pr_male_index]) >= 0.5
            val_sampler_weights[index] = 1.0 / val_subject_count[subject_id] / float(val_gender_count[is_male])

        print('Initializing val dataloader')
        val_sampler = torch.utils.data.WeightedRandomSampler(
            val_sampler_weights,
            num_samples=self.cfg['TRAIN']['VAL_BATCH_SIZE'] * self.cfg['TRAIN']['TASK_VAL_PROBE_BATCHES'])
        val_task_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.cfg['TRAIN']['VAL_BATCH_SIZE'],
            sampler=val_sampler, num_workers=self.cfg['TRAIN']['NUM_WORKERS'],
            pin_memory=bool(self.cfg['TRAIN']['PIN_MEMORY'])
        )

        return train_fc_dataloader, train_dataloader, val_task_dataloader, train_dataset.num_subjects, 2

    def _init_model_degradation(self):

        class UnitNormalize(torch.nn.Module):
            def __init__(self, init_scale, scale_as_param=False, eps=1e-10):
                super().__init__()
                self.scale = torch.tensor(init_scale, dtype=torch.float32)
                if scale_as_param:
                    self.scale = torch.nn.Parameter(self.scale)
                    self.register_parameter('scale', self.scale)
                self.eps = eps
                self.scale_as_param = scale_as_param

            def forward(self, x):
                x_unit = torch.nn.functional.normalize(x, p=2, dim=1, eps=self.eps)
                return torch.nn.functional.softplus(self.scale) * x_unit

        class FixedLinear(torch.nn.Module):
            def __init__(self, weights, bias=None):
                super().__init__()
                self.register_buffer('weights', weights)
                self.register_buffer('bias', bias)

            def forward(self, x):
                x = torch.nn.functional.linear(x, self.weights, self.bias)
                return x

        model = torch.nn.Sequential(
            torch.nn.Sequential(),  # identity placeholder
            torch.nn.Sequential(
                torch.nn.Linear(512, 256),
                torch.nn.PReLU(256)
            ))

        return model

    def _init_model_task(self, num_classes):
        model = torch.nn.Sequential(
            torch.nn.Linear(256, num_classes)
        )
        return model

    def _init_model_budget(self, num_classes):
        return budget_model.BudgetModel(
            self.cfg['MODEL']['BUDGET_INPUT_CHANNELS'],
            num_classes=num_classes,
            num_models=self.cfg['MODEL']['BUDGET_NUM_MODELS'])
