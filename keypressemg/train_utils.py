import logging
from pathlib import Path
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
import wandb


def config_wandb(args):
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="key_press_emg_toronto",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "app_name": args.app_name
        },
    )
    return run


def get_sweep_config():
    sweep_configuration = {
        "name": "sweep_key_press_emg",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "eval_acc"},
        "parameters": {
            "learning_rate": {"values": [0.001, 0.0001]},
            "weight_decay": {"values": [0, 1e-4, 1e-3, 1e-2, 1e-1]},
            "momentum": {"values": [0, 0.8, 0.9]},
            "batch_size": {"values": [64, 128]},
            "encoder_depth": {"values": [1, 2, 3]},

            # "normalize": {"values": [True, False]}
            # "learning_rate": {"min": 1e-4, "max": 0.6},
            # "high_band": {"min": 15, "max": 40},
            # "low_band": {"min": 350, "max": 500},
            # "low_pass": {"min": 7, "max": 12},

            # "encoder_activations": {"values": ["relu", "lrelu"]},
            # "encoder_depth": {"values": [1, 2, 3]},
            # "encoder_pools": {"values": ["avg_pool", "max_pool"]}
        },
    }

    return sweep_configuration


def init_sweep(config):
    sweep_id = wandb.sweep(sweep=config, project="key_press_emg_toronto")
    return sweep_id


def start_sweep(sweep_id, f_sweep):
    wandb.agent(sweep_id=sweep_id, function=f_sweep)


def get_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    # return torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return torch.optim.SGD(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay,
                           momentum=args.momentum)


def get_device(args):
    return 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'


def train(args, model, train_loader, eval_loader, optimizer, log_prefix):
    logger = logging.getLogger(args.app_name)
    device = get_device(args)
    num_epochs = args.num_epochs
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    best_acc = 0
    best_epoch = 0
    history: list[float] = []
    logger.info(f'{log_prefix}: Starting training for {num_epochs} epochs on device {device}')
    for epoch in range(num_epochs):
        epoch_loss: float = 0.0
        pbar = tqdm(train_loader)
        model.train()
        for data, target in pbar:
            batch_size: int = len(data)
            target = target.reshape(1, -1) if batch_size == 1 else target
            data = data[0] if batch_size == 1 else data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = criterion(output, target)
            loss = criterion(output, torch.nn.functional.one_hot(target, num_classes=26).float().reshape(output.shape))
            loss.backward()
            optimizer.step()
            batch_loss: float = float(loss) / batch_size
            epoch_loss += (batch_loss / float(len(train_loader)))
            pbar.set_description(
                f'{log_prefix}: Epoch {epoch + 1}/{num_epochs} batch loss: {batch_loss:.4f} epoch loss {epoch_loss:.4f}')
        logger.info(f'{log_prefix}: Epoch {epoch} loss: {epoch_loss:.4f}')
        acc, loss, sk_acc, sk_f1, _, _ = evaluate(model, eval_loader, torch.device(device))
        logger.info(f'{log_prefix}: Epoch {epoch} evaluation acc: {acc:.4f}, sk_acc {sk_acc:.4f} sk_f1 {sk_f1:.4f}')

        history.append(sk_acc)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            if args.saved_models_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'acc': acc,
                }, f'{args.saved_models_path}/{log_prefix}_epoch_{epoch + 1}_acc_{acc}.pth')

        wandb.log({f'{log_prefix}_eval_acc': acc,
                   f'{log_prefix}_eval_loss': loss,
                   f'{log_prefix}_best_acc': best_acc,
                   f'{log_prefix}_train_loss': epoch_loss,
                   f'{log_prefix}_best_epoch': best_epoch})
        logger.info(
            f'{log_prefix}_Epoch {epoch} eval acc: {acc:.4f} (best so far {best_acc: .4f} in epoch {best_epoch})'
            f' eval loss: {loss:.4f}')
    return best_acc, history


@torch.no_grad()
def evaluate(model, test_loader, device=torch.device('cpu')):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, eval_loss = 0.0, 0.0, 0.0
    model.eval()
    accs = []
    f1s = []
    all_targets = []
    all_preds = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        batch_size: int = len(data)
        target = target.reshape(1, -1) if batch_size == 1 else target
        data = data[0] if batch_size == 1 else data
        output = model(data)
        loss = criterion(output,
                         torch.nn.functional.one_hot(target, num_classes=26).float().reshape(output.shape)).item()
        eval_loss += float(loss)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy().reshape(predicted.shape)
        all_targets += target.tolist()
        all_preds += predicted.tolist()

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    balanced_score = metrics.balanced_accuracy_score(y_true, y_pred)

    total_confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    acc_score = metrics.accuracy_score(y_true, y_pred)
    eval_accuracy = (y_true == y_pred).sum().item() / total
    eval_loss /= total
    return eval_accuracy, eval_loss, acc_score, f1, total_confusion_matrix, balanced_score


def load_checkpoint(args):
    checkpoint = torch.load(Path(args.saved_models_path) / args.checkpoint)
    return checkpoint
