import torch
import argparse

from torch.utils.data import DataLoader

from nets.PAAD import PAAD
from custom_dataset import InterventionDataset
from utils import model_evaluation, density_estimation, get_F1_measure

def test_all(args):

    test_set = InterventionDataset(args.test_image_path, args.test_csv_path, 'test')
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    paad = PAAD(
        device=device,
        freeze_features=args.freeze_features,
        pretrained_file=args.pretrained_file,
        horizon=args.horizon).to(device)

    PATH = './nets/paad.pth'
    paad.load_state_dict(torch.load(PATH))

    ap = model_evaluation(test_loader, paad, device)
    print("Average precision on the test set: {:.4f}".format(ap))

    '''
    # compute density estimation on test set
    density_estimation(test_loader, paad, device, threshold=0.5)
    '''

    
    # compute F1 measure on test set
    f1_measure = get_F1_measure(test_loader, paad, device, threshold=0.5)
    print("F1 measure on the test set: {:.4f}".format(f1_measure))
    

def test_datapoint(args):

    test_set = InterventionDataset(args.test_image_path, args.test_csv_path, 'test')

    paad = PAAD(
        device='cpu',
        freeze_features=args.freeze_features,
        pretrained_file=args.pretrained_file,
        horizon=args.horizon)

    PATH = './nets/paad.pth'
    paad.load_state_dict(torch.load(PATH))

    paad.eval()

    with torch.no_grad():

        img, pred_traj_img, lidar_scan, label = test_set[args.data_index]

        img.unsqueeze_(0)
        pred_traj_img.unsqueeze_(0)
        lidar_scan.unsqueeze_(0)

        _, _, _, pred_score = paad(img, pred_traj_img, lidar_scan)

        pred_score.squeeze_()

    print("The ground truth label:", list(label.numpy().round(2)))
    print("The predicted score:   ", list(pred_score.numpy().round(2)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--test_image_path", type=str, default='test_set/images_test/')
    parser.add_argument("--test_csv_path", type=str, default='test_set/labeled_data_test.csv')
    parser.add_argument("--data_index", type=int, default=5313)

    # model parameters
    parser.add_argument("--freeze_features", type=bool, default=True)
    parser.add_argument("--pretrained_file", type=str,
                                             default="nets/VisionNavNet_state_hd.pth.tar")
    parser.add_argument("--horizon", type=int, default=10)

    args = parser.parse_args()

    #test_datapoint(args)
    test_all(args)