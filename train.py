import copy
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from nets.PAAD import PAAD
from utils import loss_fn, model_evaluation
from custom_dataset import InterventionDataset

def main(args):

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = InterventionDataset(args.train_image_path, args.train_csv_path, 'train')
    test_set  = InterventionDataset(args.test_image_path, args.test_csv_path, 'test')

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    print("Dataset is ready. Start training...")

    paad = PAAD(
        device=device,
        freeze_features=args.freeze_features,
        pretrained_file=args.pretrained_file,
        horizon=args.horizon).to(device)

    parameters = filter(lambda p: p.requires_grad, paad.parameters())
    optimizer  = torch.optim.Adam(parameters, lr=args.learning_rate,
                                              weight_decay=args.weight_decay)

    alpha = 0.01 * len(train_loader)

    # start training
    best_ap                = 0.0
    train_loss_over_epochs = []
    test_ap_over_epochs     = []

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    for epoch in range(args.epochs):

        #print("Current learning rate: ", optimizer.param_groups[0]['lr'])

        paad.train()
        running_loss = 0.0
        for iteration, (img, pred_traj_img, lidar_scan, label) in enumerate(train_loader):

            img, pred_traj_img = img.to(device), pred_traj_img.to(device)
            lidar_scan, label  = lidar_scan.to(device), label.to(device)

            recon_lidar, mean, log_var, pred_inv_score = paad(img, pred_traj_img, lidar_scan)

            loss = loss_fn(recon_lidar, lidar_scan, mean, log_var, pred_inv_score, label, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        print("Epoch {:02d}/{:02d}, Loss {:9.4f}".format(
                epoch+1, args.epochs, running_loss))

        # evaluate the model on the test set
        ap = model_evaluation(test_loader, paad, device)
        print("Average precision on the test set: {:.4f}".format(ap))

        train_loss_over_epochs.append(running_loss)
        test_ap_over_epochs.append(ap)

        # save the best model
        if ap > best_ap:
            PATH = './nets/paad.pth'
            torch.save(paad.state_dict(), PATH)
            best_ap = copy.deepcopy(ap)

        #scheduler.step()

    # plot training set loss and test set average precision over epochs
    fig = plt.figure()

    plt.subplot(2,1,1)
    plt.ylabel("Train loss")
    plt.plot(np.arange(args.epochs)+1, train_loss_over_epochs, 'k-')
    plt.title("Train loss and test average precision")
    plt.xlim(1, args.epochs)
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.ylabel("Test average precision")
    plt.plot(np.arange(args.epochs)+1, test_ap_over_epochs, 'b-')
    plt.xlabel("Epochs")
    plt.xlim(1, args.epochs)
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.close(fig)

    print("Finished training.")
    print("The best test average precision: {:.4f}".format(best_ap))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument("--train_image_path", type=str, default='train_set/images_train/')
    parser.add_argument("--train_csv_path", type=str, default='train_set/labeled_data_train.csv')
    parser.add_argument("--test_image_path", type=str, default='test_set/images_test/')
    parser.add_argument("--test_csv_path", type=str, default='test_set/labeled_data_test.csv')

    # training parameters
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.00015)
    parser.add_argument("--test_batch_size", type=int, default=64)

    # model parameters
    parser.add_argument("--freeze_features", type=bool, default=True)
    parser.add_argument("--pretrained_file", type=str,
                                             default="nets/VisionNavNet_state_hd.pth.tar")
    parser.add_argument("--horizon", type=int, default=10)

    args = parser.parse_args()

    main(args)