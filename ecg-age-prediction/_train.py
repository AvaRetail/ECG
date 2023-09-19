from tqdm import tqdm
from argparse import ArgumentParser
from dataset import ECGSequence
import sys
import torch
from resnet import ResNet1d
import logging
import time
from torcheval.metrics import MulticlassF1Score

from pathlib import Path

sys.path.append(r"C:\Users\ATI-G2\Documents\python\ECG")
from utils.pytorch_utils import SaveBestModel, save_model
from utils.training_utils import increment_path

# Finding the log file to save
save_dir = increment_path(Path("weights\lovakant\exp") , mkdir=True)  # increment run


#LOGGING 

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s ")

file_handler = logging.FileHandler(f"{save_dir}/logs.txt",mode="w")
file_logger = logging.getLogger("file")
file_logger.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
file_logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_logger = logging.getLogger("console")
console_logger.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
console_logger.addHandler(console_handler)

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ", 
#                     handlers=[file_handler, screen_handler] 
#                     )
console_logger.info(f"saving weights and logs to the folder: {save_dir}")
file_logger.info(f"saving weights and logs to the folder: {save_dir}")

def get_args():
    parser = ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=3000,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[3000, 1000, 250, 50, 10],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--dataset_name', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--val_split', type=float, default=0.02)
    parser.add_argument('path_to_hdf5',
                        help='path to file containing ECG traces',
                        default=r"12-lead.hdf5")
    parser.add_argument('path_to_csv',
                        help='path to csv file containing attributes.',
                        default=r"12-lead_labels.csv")
    args = parser.parse_args()

    return args

def setup(args):
    device = torch.device("cuda")
    N_LEADS = 12
    N_CLASSES = 63
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
    
    def model_init(m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

        if isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, 0.0,0.02)
            torch.nn.init.constant_(m.bias, 0.0)   

    model = model.apply(model_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)     

    loss = torch.nn.BCEWithLogitsLoss()

    return device, loss, optimizer, model.to(device=device)

def main():
    batch_size = 32
    train_seq, val_seq = ECGSequence.get_train_and_val(
    args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split)

    device, criterion, optim, model = setup(args)

    n_epochs = 10
    k=0

    save_best_model = SaveBestModel()

    for i in range(n_epochs):

        train_loss = []
        val_loss = []
        # train_acc = 0
        # val_acc = 0
        k = 0
        train_f1 = MulticlassF1Score(num_classes=63).to(device=device)
        val_f1 = MulticlassF1Score(num_classes=63).to(device=device)

        with tqdm(train_seq, unit="batch") as train_bar:
            # for signals, labels in train_seq:
            for signals, labels in train_bar:
                try:
                    signals, labels = signals.to(device), labels.to(device)

                    # Check if there are any non-finite values in the output
                    if not torch.isfinite(signals).all():
                        signals = torch.nan_to_num(signals)
                        
                    optim.zero_grad()
                    probs = model(signals)
                    loss = criterion(probs, labels)
                    loss.backward()
                    optim.step()

                    # loss, acc calculation
                    # _, pred_idx = torch.max(probs.data, 1)
                    _, true_idx = torch.max(labels.data, 1)
                    # train_running_correct += (pred_idx==true_idx).sum().item()
                    train_loss.append(loss.item())

                    train_f1.update(probs, true_idx)
                    train_score = train_f1.compute()

                    k = k+ 1

                    train_bar.set_postfix(iteration= f"{k+1}/{len(train_seq)}", loss = loss.item(), f1_score = train_score.item())

                except Exception as e:
                    file_logger.error(e)
                    # console_handler.error(e)

            train_epoch_loss = sum(train_loss)/len(train_loss)
            file_logger.info(f"training f1 score for the epoch-{i+1}/{n_epochs}: {train_score}")
            file_logger.info(f"training loss for the epoch-{i+1}/{n_epochs}: {train_epoch_loss}")

            console_logger.info(f"training f1 score for the epoch-{i+1}/{n_epochs}: {train_score}")
            console_logger.info(f"training loss for the epoch-{i+1}/{n_epochs}: {train_epoch_loss}")
    
        with tqdm(val_seq) as val_bar:
            with torch.no_grad():

                for signals, labels in val_bar:
                    try:
                        signals, labels = signals.to(device), labels.to(device)

                        # Checking if there are any non-finite values in the output
                        if not torch.isfinite(signals).all():
                            signals = torch.nan_to_num(signals)
                        probs = model(signals)
                        loss = criterion(probs, labels)

                        # loss and metric calculation
                        _, true_idx = torch.max(labels, 1)
                        val_loss.append(loss.item())
                        val_f1.update(probs, true_idx)
                        val_score = val_f1.compute()

                        #scripting into files and std outuput
                        val_bar.set_postfix(val_loss=loss.item(), val_f1 = val_score.item())

                    except Exception as e:
                        file_logger.error(e)

            val_epoch_loss = sum(val_loss)/len(val_loss)
            file_logger.info(f"val f1 score for the epoch-{i+1}/{n_epochs}: {val_score}")
            file_logger.info(f"val loss for the epoch-{i+1}/{n_epochs}: {val_epoch_loss}")
    
            console_logger.info(f"val f1 score for the epoch-{i+1}/{n_epochs}: {val_score}")
            console_logger.info(f"val loss for the epoch-{i+1}/{n_epochs}: {val_epoch_loss}")

            save_best_model(val_epoch_loss, i, model, optim, criterion, save_dir)
    
    save_model(n_epochs, model, optim, criterion, save_dir)


if __name__=="__main__":
    args = get_args()
    main()

