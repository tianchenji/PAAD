import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.metrics import average_precision_score, confusion_matrix

#plt.rcParams.update({
#    'font.size': 24
#})

def loss_fn(recon_x, x, mean, log_var, pred_inv_score, y, alpha):

    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    CLF = torch.nn.functional.binary_cross_entropy(pred_inv_score, y, reduction='sum')

    return (BCE + KLD + alpha * CLF) / x.size(0)

def model_evaluation(data_loader, model, device):

    gt_list         = []
    pred_score_list = []
    
    model.eval()

    with torch.no_grad():
        for (img, pred_traj_img, lidar_scan, label) in data_loader:
            img, pred_traj_img  = img.to(device), pred_traj_img.to(device)
            lidar_scan          = lidar_scan.to(device)
            _, _, _, pred_score = model(img, pred_traj_img, lidar_scan)

            gt_list.extend(list(label.flatten()))
            pred_score_list.extend(list(pred_score.cpu().numpy().flatten()))

    return average_precision_score(gt_list, pred_score_list)

def get_F1_measure(data_loader, model, device, threshold):

    gt_list         = []
    pred_label_list = []

    model.eval()

    with torch.no_grad():
        for (img, pred_traj_img, lidar_scan, label) in data_loader:
            img, pred_traj_img  = img.to(device), pred_traj_img.to(device)
            lidar_scan          = lidar_scan.to(device)
            _, _, _, pred_score = model(img, pred_traj_img, lidar_scan)

            pred_label = pred_score > threshold

            gt_list.extend(list(label.flatten()))
            pred_label_list.extend(list(pred_label.cpu().numpy().flatten()))

    tn, fp, fn, tp = confusion_matrix(gt_list, pred_label_list).ravel()
    precision      = tp / (tp + fp)
    recall         = tp / (tp + fn)
    f1_measure     = (2 * precision * recall) / (precision + recall)

    return f1_measure

def logit(x):
    return np.log(x / (1 - x))

def logistic(x):
    return np.exp(x) / (1 + np.exp(x))

def density_estimation(data_loader, model, device, threshold=None):

    normal_score  = []
    failure_score = []

    model.eval()

    with torch.no_grad():
        for (img, pred_traj_img, lidar_scan, label) in data_loader:
            img, pred_traj_img  = img.to(device), pred_traj_img.to(device)
            lidar_scan          = lidar_scan.to(device)
            _, _, _, pred_score = model(img, pred_traj_img, lidar_scan)

            label      = label.flatten()
            pred_score = pred_score.cpu().numpy().flatten()

            for label_i, pred_score_i in zip(label, pred_score):
                if label_i == 0:
                    normal_score.append([pred_score_i])
                else:
                    failure_score.append([pred_score_i])

    normal_score  = np.array(normal_score)
    failure_score = np.array(failure_score)

    normal_score = np.clip(normal_score, a_min=1e-6, a_max=1-1e-6)
    failure_score = np.clip(failure_score, a_min=1e-6, a_max=1-1e-6)

    # transformation trick
    normal_score_tf, failure_score_tf = logit(normal_score), logit(failure_score)

    kde_normal  = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(normal_score_tf)
    kde_failure = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(failure_score_tf)
    X_plot    = np.linspace(0.001, 0.999, 1000)[:, np.newaxis]
    X_plot_tf = logit(X_plot)
    log_dens_normal_tf  = kde_normal.score_samples(X_plot_tf)
    log_dens_failure_tf = kde_failure.score_samples(X_plot_tf)

    dens_normal  = np.exp(log_dens_normal_tf)[:, np.newaxis] / (X_plot * (1 - X_plot))
    dens_failure = np.exp(log_dens_failure_tf)[:, np.newaxis] / (X_plot * (1 - X_plot))

    if not threshold:
        plt.figure(figsize=(10, 9))
        plt.fill_between(X_plot[:, 0], dens_normal[:, 0], fc='#AAAAFF', alpha=0.3, label='normal')
        plt.fill_between(X_plot[:, 0], dens_failure[:, 0], fc='#FFAAAA', alpha=0.3, label='failure')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        plt.axis([0, 1, 0, 1.5])
        plt.show()
    else:
        plt.figure(figsize=(10, 9))
        plt.fill_between(X_plot[:, 0], dens_normal[:, 0], fc='#0066CC', alpha=0.3)
        plt.fill_between(X_plot[:, 0], dens_failure[:, 0], fc='#FA6C00', alpha=0.3)
        plt.plot(X_plot[:, 0], dens_normal[:, 0], color='#0066CC', linewidth=3.0, label='normal')
        plt.plot(X_plot[:, 0], dens_failure[:, 0], color='#FA6C00', linewidth=3.0, label='failure')
        plt.axvline(x=threshold, ymin=0, ymax=1,
                    color='black', ls='--', linewidth=2.0, label='threshold')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.axis([0, 1, 0, 1.5])
        plt.show()