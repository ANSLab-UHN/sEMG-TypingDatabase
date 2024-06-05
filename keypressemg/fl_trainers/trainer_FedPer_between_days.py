import argparse
import copy
import logging
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import wandb
from sklearn import metrics
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

from keypressemg.common.folder_paths import DATA_ROOT
from keypressemg.common.types_defined import Participant, DayT1T2
from keypressemg.datasets.split_between_days_dataset import get_split_between_days_dataset
from keypressemg.fl_trainers.utils import get_device, set_logger, set_seed, str2bool
from keypressemg.models.feature_model import FeatureModel


def get_personalization_net(args) -> nn.Module:
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
    net = nn.Sequential(*[nn.Linear(in_features=args.num_classes, out_features=args.num_classes)
                          for _ in range(args.num_personalization_layers)])
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
    net.to(device)
    return net


class PersonalizedModel(nn.Module):
    def __init__(self, global_net: nn.Module, personalization_net: nn.Module):
        super().__init__()
        self._global_net = global_net
        self._personalization_layer = personalization_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._global_net(x)
        x = self._personalization_layer(x)
        return x

    def get_global_params(self, params: OrderedDict) -> OrderedDict:
        for n, p in self._global_net.named_parameters():
            params[n] += p.data
        return params


def get_optimizer(args, network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


@torch.no_grad()
def eval_model(args, global_model, client_ids, clients_personalization_nets):
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

    loss_dict: dict[str, float] = {}
    acc_dict: dict[str, float] = {}
    acc_score_dict: dict[str, float] = {}
    f1s_dict: dict[str, float] = {}
    criteria = torch.nn.CrossEntropyLoss()

    y_true_all, y_pred_all, loss_all = None, None, 0.

    # global_model.eval()
    num_clients = len(client_ids)

    for i, client_id in enumerate(client_ids):
        local_net = PersonalizedModel(global_net=global_model,
                                      personalization_net=clients_personalization_nets[client_id])
        local_net.eval()

        running_loss, running_correct, running_samples = 0., 0., 0.

        p = Participant(f"P{client_id}")
        t = DayT1T2.T1

        _, eval_set = get_split_between_days_dataset(Path(args.data_path), p, t, scale=True)
        test_loader = DataLoader(eval_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

        all_targets = []
        all_preds = []

        for batch_count, batch in enumerate(test_loader):
            X_test, Y_test = tuple(t.to(device) for t in batch)
            pred = local_net(X_test)
            # pred = global_model(X_test)

            loss = criteria(pred, Y_test)
            predicted = torch.max(pred, dim=1)[1].cpu().numpy()

            running_loss += (loss.item() * Y_test.size(0))
            running_correct += pred.argmax(1).eq(Y_test).sum().item()
            running_samples += Y_test.size(0)

            target = Y_test.cpu().numpy().reshape(predicted.shape)

            all_targets += target.tolist()
            all_preds += predicted.tolist()

        # calculate confusion matrix
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        running_loss /= running_samples

        y_true_all = y_true if y_true_all is None else np.concatenate((y_true_all, y_true), axis=0)
        y_pred_all = y_pred if y_pred_all is None else np.concatenate((y_pred_all, y_pred), axis=0)
        loss_all += (running_loss / num_clients)

        eval_accuracy = (y_true == y_pred).sum().item() / running_samples
        acc_score = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='micro')

        acc_dict[f"P{client_id}"] = eval_accuracy
        loss_dict[f"P{client_id}"] = running_loss
        acc_score_dict[f"P{client_id}"] = acc_score
        f1s_dict[f"P{client_id}"] = f1

    avg_acc = (y_true_all == y_pred_all).mean().item()
    avg_loss = loss_all
    avg_acc_score = metrics.accuracy_score(y_true_all, y_pred_all)
    avg_f1 = metrics.f1_score(y_true_all, y_pred_all, average='micro')

    return acc_dict, loss_dict, acc_score_dict, f1s_dict, avg_acc, avg_loss, avg_acc_score, avg_f1


def train(args):
    client_ids = [int(p.value.replace('P', '')) for p in Participant]
    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
    clients_personalization_nets = {cid: get_personalization_net(args) for cid in client_ids}

    net = FeatureModel(cls_layer=True)
    net = net.to(device)
    criteria = torch.nn.CrossEntropyLoss()

    best_acc, best_epoch, best_loss, best_acc_score, best_f1 = 0., 0., 0., 0., 0.
    step_iter = trange(args.num_steps)

    for step in step_iter:

        # select several clients
        client_ids_step = np.random.choice(client_ids, size=args.num_client_agg, replace=False)

        # initialize global model params
        params = OrderedDict()
        for n, p in net.named_parameters():
            params[n] = torch.zeros_like(p.data)

        # iterate over each client
        train_avg_loss = 0
        num_samples = 0

        for j, c_id in enumerate(client_ids_step):

            curr_global_net = copy.deepcopy(net)
            local_net = PersonalizedModel(global_net=curr_global_net,
                                          personalization_net=clients_personalization_nets[c_id])
            local_net.train()
            optimizer = get_optimizer(args, local_net)
            # curr_global_net.train()
            # optimizer = get_optimizer(args, curr_global_net)

            p = Participant(f"P{c_id}")
            t = DayT1T2.T1

            train_set, _ = get_split_between_days_dataset(Path(args.data_path), p, t, scale=True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.num_workers)

            for i in range(args.inner_steps):
                batch = next(iter(train_loader))
                x, Y = tuple(t.to(device) for t in batch)

                optimizer.zero_grad()
                pred = local_net(x)
                # pred = curr_global_net(x)

                loss = criteria(pred, Y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_net.parameters(), 50)
                # torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
                optimizer.step()

                train_avg_loss += loss.item() * Y.shape[0]
                num_samples += Y.shape[0]

                step_iter.set_description(
                    f"Step: {step + 1}, client: {c_id}, Inner Step: {i}, Loss: {loss.item()}"
                )

            params: OrderedDict = local_net.get_global_params(params)
            # for n, p in curr_global_net.named_parameters():
            #     params[n] += p.data

        train_avg_loss /= num_samples

        # average parameters
        for n, p in params.items():
            params[n] = p / args.num_client_agg
        # update new parameters
        net.load_state_dict(params)

        if step % args.eval_every == 0 or (step + 1) == args.num_steps:
            test_acc_dict, test_loss_dict, test_acc_score_dict, test_f1s_dict, \
                test_avg_acc, test_avg_loss, test_avg_acc_score, test_avg_f1 = eval_model(args, net, client_ids,
                                                                                          clients_personalization_nets)

            if test_avg_acc > best_acc:
                best_acc = test_avg_acc
                best_loss = test_avg_loss
                best_acc_score = test_avg_acc_score
                best_f1 = test_avg_f1
                best_epoch = step

        if args.wandb:
            log_dict = {}
            log_dict.update(
                {
                    'custom_step': step,
                    'train_loss': train_avg_loss,
                    'test_avg_loss': test_avg_loss,
                    'test_avg_acc': test_avg_acc,
                    'test_avg_acc_score': test_avg_acc_score,
                    'test_avg_f1': test_avg_f1,
                    'test_best_loss': best_loss,
                    'test_best_acc': best_acc,
                    'test_best_acc_score': best_acc_score,
                    'test_best_f1': best_f1,
                    'test_best_epoch': best_epoch
                }
            )
            log_dict.update({f"test_acc_{l}": m for (l, m) in test_acc_dict.items()})
            log_dict.update({f"test_loss_{l}": m for (l, m) in test_loss_dict.items()})
            log_dict.update({f"test_acc_score_{l}": m for (l, m) in test_acc_score_dict.items()})
            log_dict.update({f"test_f1_{l}": m for (l, m) in test_f1s_dict.items()})
            mean_acc = np.mean(list(test_acc_dict.values()))
            std_acc = np.std(list(test_acc_dict.values()))
            log_dict.update({'mean_acc': mean_acc})
            log_dict.update({'std_acc': std_acc})

            wandb.log(log_dict)

        logging.info(
            f"epoch {step}, "
            f"train loss {train_avg_loss:.3f}, "
            f"best epoch: {best_epoch:.3f}, "  # , lr={lr}
            f"best test loss: {best_loss:.3f}, "  # , lr={lr}
            f"best test acc: {best_acc:.3f}, "
            f"test test acc score: {best_acc_score:.3f}, "
            f"best test f1: {best_f1:.3f}"
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Personalized Federated Learning")

    #############################
    #       Data args        #
    #############################
    parser.add_argument("--data-path", type=str,
                        default=f'{DATA_ROOT.as_posix()}/valid_user_features',
                        help="dir path for datafolder")
    parser.add_argument("--app-name", type=str, default=f"train_FedPer_mlp_between_days")
    parser.add_argument("--num-classes", type=int, default=26, help="Number of unique labels")

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inner-steps", type=int, default=10, help="number of inner steps")
    parser.add_argument("--num-client-agg", type=int, default=5, help="number of cleints per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--num-personalization-layers", type=float, default=1,
                        help="Number of linear layers each client uses for personalization")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
    parser.add_argument("--eval-every", type=int, default=20, help="eval every X selected steps")
    parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument('--wandb', type=str2bool, default=False)

    args = parser.parse_args()

    set_logger()
    set_seed(args.seed)

    exp_name = f'FedPer_between-days_seed_{args.seed}_wd_{args.wd}_' \
               f'lr_{args.lr}_num-steps_{args.num_steps}_inner-steps_{args.inner_steps}'

    # Weights & Biases
    if args.wandb:
        wandb.init(project="key_press_emg_toronto", name=exp_name)
        wandb.config.update(args)

    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
    train(args)
