import torch
from argparse import ArgumentParser
import os
from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC

# importing functions from withing dir
from dataset import ECGSequence
from resnet import ResNet1d


def main():
    _, _, test_data = ECGSequence.get_train_and_val(args.path_to_hdf5,
                                  args.dataset_name,
                                  args.path_to_csv,
                                  batch_size=32,
                                  split=0.1)
    
    device = torch.device("cuda")
    N_LEADS = 12
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
    ckpt = torch.load(args.pre_trained_pth, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    acc = MulticlassAccuracy(num_classes=args.n_classes)

    for i, (data, labels) in enumerate(test_data):
        data, labels = data.to(device=device), labels.to(device=device)
        logits = model(data)
        # probs = torch.nn.functional(logits)
        _, true_idx = torch.max(labels, 1)

        acc.update(logits, true_idx)
        # metric = acc.compute()
    print("Your model is giving the accuracy of: ", acc.compute().item())



def get_args():

    parser = ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--dataset_name', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--path_to_hdf5',
                        help='path to file containing ECG traces',
                        default=r"E:\Chetan\ECG\data\lovakant\exp4\12-lead.hdf5")
    parser.add_argument('--path_to_csv',
                        help='path to csv file containing attributes.',
                        default=r"E:\Chetan\ECG\data\lovakant\exp4\12-lead-labels.csv")
    parser.add_argument("--pre_trained_pth", 
                        type=str, 
                        default=os.path.join(r"E:\Chetan\ECG\weights\lovakant\exp6", "best_model.pth"))
    
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    main()
