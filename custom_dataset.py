import os
import sys
import csv
import torch
import numpy as np

from skimage import io
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset

class InterventionDataset(Dataset):
    def __init__(self, image_path, csv_path, dataset_type):
        self.samples      = []
        self.image_path   = image_path
        self.csv_path     = csv_path
        self.dataset_type = dataset_type

        # parameters for the ego view
        # image size in pixels
        self.w_ego, self.h_ego = 320, 240
        # focal length in pixels
        self.f = 460
        # camera height in meters
        self.Y = - 0.23

        self.lidar_clip = 1.85

        self.normal_failure_ratio = self.compute_ratio() - 1

        self.read_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, pred_traj, lidar_scan, label = self.samples[idx]

        # apply color jitter to the training set
        if self.dataset_type == 'train':
            color_jitter_transform = transforms.RandomApply(
                        [transforms.ColorJitter(0.5, 0.25, 0.25, 0.1)], p=0.5)
            image = color_jitter_transform(image)

        # normalize the image
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = image_transform(image)

        # draw the predicted trajectory
        pred_traj_img = Image.new(mode="L", size=(self.w_ego, self.h_ego))
        traj_draw     = ImageDraw.Draw(pred_traj_img)
        traj_draw.line(pred_traj, fill="white", width=6, joint="curve")

        # use the region of interest
        pred_traj_img = pred_traj_img.crop((0, 112, 320, 240))

        pred_traj_transform = transforms.Compose([transforms.ToTensor()])
        pred_traj_img = pred_traj_transform(pred_traj_img)

        return (image, pred_traj_img, lidar_scan, label)

    def compute_ratio(self):

        num_of_normal  = 0
        num_of_failure = 0
        map_int = lambda x: np.array(list(map(int, x)))

        with open(self.csv_path, newline='') as data_csv:
            data_reader = csv.DictReader(data_csv)
            for datapoint in data_reader:
                label = datapoint['label'][1:-1].split(',')
                label = map_int(label)

                # count the number of normal cases
                if max(label) == 0:
                    num_of_normal += 1
                else:
                    num_of_failure += 1

        # include flipped images
        num_of_failure = num_of_failure * 2
        nf_ratio = round(num_of_normal / num_of_failure)

        if self.dataset_type == 'train':
            print("The ratio of normal to augmented failure is: {:d}".format(nf_ratio))

        return nf_ratio

    def get_pred_traj_ego(self, pred_Zs, pred_Xs):
        '''
        compute the predicted trajectory in the image coordinate
        '''

        # project 3D points onto the image plane
        xs = self.f * pred_Xs / pred_Zs
        ys = self.f * self.Y / pred_Zs

        # convert 2D points into the image coordinate
        xs = - xs + self.w_ego / 2
        ys = - ys + self.h_ego / 2

        pred_traj = [(x, y) for x, y in zip(xs, ys)]

        return pred_traj

    def read_datapoint(self, datapoint):

        map_float = lambda x: np.array(list(map(float, x)))
        map_int   = lambda x: np.array(list(map(int, x)))

        img_name = os.path.join(self.image_path, datapoint['image_name'])
        image    = io.imread(img_name)
        image    = Image.fromarray(image)

        pred_Zs   = datapoint['pred_traj_x'][1:-1].split(',')
        pred_Xs   = datapoint['pred_traj_y'][1:-1].split(',')
        pred_Zs   = abs(map_float(pred_Zs))
        pred_Xs   = map_float(pred_Xs)

        lidar_scan = datapoint['lidar_scan'][1:-1].split(',')
        lidar_scan = map_float(lidar_scan)
        lidar_scan = np.clip(lidar_scan, a_min=0, a_max=self.lidar_clip)/self.lidar_clip
        lidar_scan = torch.as_tensor(lidar_scan, dtype=torch.float32)

        label = datapoint['label'][1:-1].split(',')
        label = map_int(label)
        label = torch.as_tensor(label, dtype=torch.float32)

        return (image, pred_Zs, pred_Xs, lidar_scan, label)

    def read_data(self):

        num_of_normal = 0
        print_every = print_threshold = 2000

        data_counter     = [0, 0] # [normal, failure]
        data_counter_aug = [0, 0] # [normal, failure]

        with open(self.csv_path, newline='') as data_csv:
            data_reader = csv.DictReader(data_csv)
            for datapoint in data_reader:
                image, pred_Zs, pred_Xs, lidar_scan, label = self.read_datapoint(datapoint)
                pred_traj = self.get_pred_traj_ego(pred_Zs, pred_Xs)

                if self.dataset_type == 'train':
                    if max(label) == 0:
                        data_counter[0] += 1
                        # under-sample normal cases
                        if num_of_normal % self.normal_failure_ratio == 0:
                            self.samples.append([image, pred_traj, lidar_scan, label])
                            data_counter_aug[0] += 1
                        num_of_normal += 1

                    else:
                        data_counter[1] += 1
                        self.samples.append([image, pred_traj, lidar_scan, label])
                        data_counter_aug[1] += 1

                        # augment the training set by flipping images of failures
                        image_flipped      = transforms.functional.hflip(image)
                        pred_traj_flipped  = self.get_pred_traj_ego(pred_Zs, - pred_Xs)
                        lidar_scan_flipped = torch.flip(lidar_scan, [0])
                        self.samples.append([image_flipped, pred_traj_flipped,
                                             lidar_scan_flipped, label])
                        data_counter_aug[1] += 1
                
                else:
                    self.samples.append([image, pred_traj, lidar_scan, label])
                    if max(label) == 0:
                        data_counter[0] += 1
                        data_counter_aug[0] += 1
                    else:
                        data_counter[1] += 1
                        data_counter_aug[1] += 1

                if sum(data_counter_aug) >= print_threshold:
                    print("Data loaded: {:d}".format(print_threshold))
                    print_threshold += print_every


        assert sum(data_counter_aug) == len(self.samples)

        print("All data have been loaded! Total dataset size: {:d}".format(len(self.samples)))
        print("The number of normal cases / failures: {:d} / {:d}".format(
            data_counter_aug[0], data_counter_aug[1]))
        print("The number of normal cases / failures before augmentation: {:d} / {:d}".format(
            data_counter[0], data_counter[1]))

if __name__ == '__main__':

    image_path = 'train_set/images_train/'
    csv_path   = 'train_set/labeled_data_train.csv'

    dataset = InterventionDataset(image_path, csv_path, 'train')