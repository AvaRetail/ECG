from tqdm import tqdm
from argparse import ArgumentParser
from dataset import ECGSequence
import sys
import torch
from resnet import ResNet1d
import logging
import time

sys.path.append(r"C:\Users\ATI-G2\Documents\python\ECG")

#LOGGING 

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s ")

file_handler = logging.FileHandler(f"logs.txt",mode="w")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)     

    loss = torch.nn.BCEWithLogitsLoss()

    return device, loss, optimizer, model.to(device=device)

def main():
    batch_size = 32
    train_seq, val_seq = ECGSequence.get_train_and_val(
    args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split)

    device, criterion, optim, model = setup(args)

    n_epochs = 2
    k=0

    train_loss = []
    val_loss = []

    for i in range(n_epochs):
        # for signals, labels in train_seq:
        for signals, labels in tqdm(train_seq, f'epoch-{i+1}/{n_epochs}'):
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
                train_loss.append(loss.item())
                console_handler.info(loss.item())
                console_handler.debug(f"{k}")
                k = k+ 1

            except Exception as e:
                file_handler.error(e)
                # console_handler.error(e)

        file_handler.info(f"training loss for the epoch-{i+1}/{n_epochs}: {sum(train_loss)/len(train_loss)}")
        train_loss.clear()
        k=0
    
        with torch.no_grad():

            for signals, labels in tqdm(val_seq, "validating"):
                try:
                    signals, labels = signals.to(device), labels.to(device)
                    # Check if there are any non-finite values in the output
                    if not torch.isfinite(signals).all():
                        signals = torch.nan_to_num(signals)
                    probs = model(signals)
                    loss = criterion(probs, labels)
                    console_handler.info(loss.item())
                    val_loss.append(loss.item())

                except Exception as e:
                    file_handler.error(e)

            file_handler.info(f"val loss for the epoch-{i+1}/{n_epochs}: {sum(val_loss)/len(val_loss)}")
            val_loss.clear()
 
if __name__=="__main__":
    args = get_args()
    main()

