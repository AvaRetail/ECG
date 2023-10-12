import sys
import torch
import logging
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from torcheval.metrics import MulticlassAUPRC, MulticlassAUROC, MulticlassAccuracy
from pathlib import Path
from torch import nn
import os

# importing functions from withing dir
from dataset import ECGSequence
# from resnet import ResNet1d
from _resnet import resnet34

#importing functions from utils
sys.path.append(r"E:\Chetan\ECG")
from utils.pytorch_utils import SaveBestModel, save_model
from utils.utils import increment_path

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

console_logger.info(f"saving weights and logs to the folder: {save_dir}")

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ", 
#                     handlers=[file_handler, screen_handler] 
#                     )

# file_logger.info(f"saving weights and logs to the folder: {save_dir}")


def calc_f1_score(input):
    P, R, T = input[0], input[1], input[2]
    f1_score = []
    f1 = 0

    for p,r,t in zip(P,R,T): # this loops as many number of labels or classes
        Len = len(p)
        for i in range(Len):
            pi, ri = p[i].item(), r[i].item()
            f1+= 2*pi*ri/(pi+ri)
        f1_score.append(f1/Len)
    return f1_score.sum()/len(f1_score)

def setup():
    device = torch.device("cuda")

    if not args.resume_training:
        N_LEADS = 12
        # N_CLASSES = 1 # will be converted to 7
        # seq_length = 4096
        # net_filter_size = [64, 128, 196, 256, 320]
        # net_seq_length = [4096, 1024, 256, 64, 16]
        # kernel_size = 21
        # dropout_rate = 0.8

        # model = ResNet1d(input_dim=(N_LEADS, seq_length),
        #                 blocks_dim=list(zip(net_filter_size, net_seq_length)),
        #                 n_classes=N_CLASSES,
        #                 kernel_size=kernel_size,
        #                 dropout_rate=dropout_rate)
        model = resnet34(num_classes=args.n_classes, input_channels = N_LEADS)

        file_logger.info("--- tranining parameters ---")
        # file_logger.info(f"seq_length: {seq_length}")
        # file_logger.info(f"net_filter_size: {net_filter_size}")
        # file_logger.info(f"net_seq_length: {net_seq_length}")
        # file_logger.info(f"kernel_size: {kernel_size}")
        # file_logger.info(f"dropout_rate: {dropout_rate}")

        def model_init(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.normal_(m.weight, 0.0, 0.09)

            if isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.normal_(m.weight, 0.0,0.09)
                torch.nn.init.constant_(m.bias, 0.0)   

        model = model.apply(model_init)

        # model.load_state_dict(torch.load(args.pre_trained_pth)["model"])
        
        # lst = model.lin.in_features
        # model.lin = nn.Linear(lst, args.n_classes)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)     
        loss = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3)

        return device, loss, optimizer, model.to(device=device), scheduler

    else:
        N_LEADS = 12
        N_CLASSES = 7 # will be converted to 7
        seq_length = 4096
        net_filter_size = [64, 128, 196, 256, 320]
        net_seq_length = [4096, 1024, 256, 64, 16]
        kernel_size = 17
        dropout_rate = 0.8

        model = ResNet1d(input_dim=(N_LEADS, seq_length),
                        blocks_dim=list(zip(net_filter_size, net_seq_length)),
                        n_classes=args.n_classes,
                        kernel_size=kernel_size,
                        dropout_rate=dropout_rate)
        model.to(device=device)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
        loss = torch.nn.CrossEntropyLoss()

        checkpoint = torch.load(args.resume_pth, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])   
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
        # loss = checkpoint["loss"]
        # model.train()
        del checkpoint
        

        return device, loss, optimizer, model



def main():
    train_seq, val_seq, _ = ECGSequence.get_train_and_val(
    args.path_to_hdf5, args.dataset_name, args.path_to_csv, args.batch_size, args.val_split)

    device, criterion, optim, model, scheduler = setup()

    n_epochs = args.epochs
    k=0

    save_best_model = SaveBestModel()

    train_loss_current = []
    val_loss_current = []
    train_loss = []
    val_loss = []
    train_score = []
    val_score = []

    for i in range(n_epochs):

        k = 0
        F1 = 0

        # train_f1 = MulticlassAUPRC(num_classes=args.n_classes).to(device=device)
        # val_f1 = MulticlassAUPRC(num_classes=args.n_classes).to(device=device)

        train_f1 = MulticlassAccuracy(num_classes=args.n_classes)
        val_f1 = MulticlassAccuracy(num_classes=args.n_classes)
        # softmax = torch.nn.functional.softmax(dim=1)

        with tqdm(train_seq, unit="batch") as train_bar:
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

                    train_loss_current.append(round(loss.item(),5))

                    train_f1.update(torch.nn.functional.softmax(probs, dim=1), true_idx)
                    metric = train_f1.compute()

                    k = k+ 1
                    train_bar.set_postfix(iteration= f"{k+1}/{len(train_seq)}", loss = loss.item(), metric = metric.item())

                except Exception as e:
                    file_logger.error(e)

            train_epoch_loss = sum(train_loss_current)/len(train_loss_current) # calculating whole loss for the current epoch
            scheduler.step(train_epoch_loss)
            train_loss.append(train_epoch_loss)
            train_score.append(round(metric.item(),4)) # appending the metric from the current epoch to save into logs

            console_logger.info(f"training f1 score for the epoch-{i+1}/{n_epochs}: {round(metric.item(),4)}")
            console_logger.info(f"training loss for the epoch-{i+1}/{n_epochs}: {train_epoch_loss}")
    
        with tqdm(val_seq, unit="batch") as val_bar:
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
                        val_loss_current.append(round(loss.item(),5))

                        val_f1.update(probs, true_idx)
                        val_metric = val_f1.compute()

                        val_bar.set_postfix(val_loss_current=loss.item(), metric = val_metric.item())

                    except Exception as e:
                        file_logger.error(e)


            val_epoch_loss = sum(val_loss_current)/len(val_loss_current)
            val_loss.append(val_epoch_loss)
            val_score.append(round(metric.item(),4))
    
            console_logger.info(f"val f1 score for the epoch-{i+1}/{n_epochs}: {round(val_metric.item(),4)}")
            console_logger.info(f"val loss for the epoch-{i+1}/{n_epochs}: {val_epoch_loss}")

            save_best_model(val_epoch_loss, i, model, optim, loss, save_dir)
    
    save_model(n_epochs, model, optim, loss, save_dir)

    df_data = {"epoch": range(args.epochs),
               "train_loss": train_loss,
               "val_loss": val_loss,
               "train_metric": train_score,
               "val_metric": val_score}
    
    pd.DataFrame(df_data).to_csv(f"{save_dir}\logs.csv",index = False) #saving logs

def get_args():
    parser = ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument("--pre_trained_pth", type=str, default=r"models\automatic-ecg-diagnosis-pyTorch\pretrained-weights\model.pth")
    parser.add_argument("--resume_training", type=bool, default=False)
    parser.add_argument("--resume_pth", type=str, default=os.path.join(r"E:\Chetan\ECG\weights\lovakant\exp4", "final_model.pth"))
    parser.add_argument("--n_classes", type = int, default=3)
    parser.add_argument('--epochs', type=int, default=40,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help="define the number of samples per each batch")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dataset_name', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--path_to_hdf5',
                        help='path to file containing ECG traces',
                        default=r"E:\Chetan\ECG\data\lovakant\exp4\12-lead.hdf5")
    parser.add_argument('--path_to_csv',
                        help='path to csv file containing attributes.',
                        default=r"E:\Chetan\ECG\data\lovakant\exp4\12-lead-labels.csv")
    
    args = parser.parse_args()

    return args


if __name__=="__main__":
    args = get_args()
    main()

