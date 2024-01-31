import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns

from scipy.signal import butter, lfilter, medfilt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


def plot_traj_on_intersection(df, alpha=.2, plot_name=None):
    angle = 0.43 * np.pi

    flip = np.array([-1, 1])

    scale = np.array([1 / 0.03, 1 / 0.03])

    shift = np.array([770, 440])

    rotate = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])

    coordinates_list = []
    for name, group in df.groupby(level=df.index.names[:-1]):

        coordinates = np.transpose(rotate @ np.transpose(group.values))*flip*scale + shift
        coordinates_list.append(coordinates)

    im = plt.imread("./data/intersection_map.png")

    plt.figure(figsize=(10,100))
    plt.xlim((200, 1400))
    plt.ylim((1000, 0))
    implot = plt.imshow(im)
    for coordinates in tqdm(coordinates_list, desc='cooking plot'):
        plt.plot(coordinates[:, 0], coordinates[:, 1], c='royalblue', alpha=alpha)
    
    angle2 = 0.2 * np.pi
    rotate2 = np.array([[np.cos(angle2), -np.sin(angle2)],
                       [np.sin(angle2), np.cos(angle2)]])
        
    loc_x_axis_vals = np.transpose(np.vstack([np.arange(-100,100), np.zeros(200)]))
    loc_y_axis_vals = np.transpose(np.vstack([np.zeros(200), np.arange(-100,100)]))
    
    x_axis_coor = np.transpose(rotate2 @ np.transpose(loc_x_axis_vals))*np.array([-1, 1]) + shift
    y_axis_coor = np.transpose(rotate2 @ np.transpose(loc_y_axis_vals))*np.array([-1, 1]) + shift
        
    plt.plot(x_axis_coor[:, 0], x_axis_coor[:, 1], c='black')
    plt.plot(y_axis_coor[:, 0], y_axis_coor[:, 1], c='black')

    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    
    if plot_name is not None:
        plt.savefig(plot_name + '.svg')
        plt.savefig(plot_name + '.pdf')
        plt.savefig(plot_name + '.png')

    plt.show()
    
    
def plot_reconstruction_on_intersection(loader, autoencoder, scaler, z_ood_list=None, alpha_id=.1, alpha_ood=1, 
                                        name=None, id_plot=True, medfilt_kernel_size=None, device=None):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    angle = 0.43 * np.pi
    flip = np.array([-1, 1])
    scale = np.array([1 / 0.03, 1 / 0.03])
    shift = np.array([770, 440])
    rotate = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    
    ###################################################
    if id_plot:
        recons_list = []
        for x_, _ in tqdm(loader, desc='decoding batches'):
            x_ = x_.to(device)
            
            try:
                mu, var = autoencoder.encode(x_)
                z = autoencoder.reparameterize(mu, var)
            except:
                z = autoencoder.encode(x_)
            
            rec_ = autoencoder.decode(z)
            recons_list.append(rec_.cpu().detach())
        recons = np.vstack(recons_list)

        coordinates_list = []
        for recon in recons:
            coordinates = np.transpose(rotate @ np.transpose(scaler.inverse_transform(recon)))*flip*scale + shift
            coordinates_list.append(coordinates)
        
    ###################################################
    if z_ood_list is not None:
        stacked_recon = []
        for z_ood in z_ood_list:
            recon_ood_list = []
            for z_i in z_ood:
                z_i = torch.tensor(z_i).float().cuda()
                z_i = z_i.view(1, -1)
                
                recon = autoencoder.decode(z_i).cpu().detach()
                coordinates_ood = (scaler.inverse_transform(np.array(recon)) @ rotate)*flip*scale + shift
                coordinates_ood = coordinates_ood.squeeze()
                
                if medfilt_kernel_size is not None:
                    coordinates_ood[:, 0] = medfilt(coordinates_ood[:, 0], kernel_size=medfilt_kernel_size)
                    coordinates_ood[:, 1] = medfilt(coordinates_ood[:, 1], kernel_size=medfilt_kernel_size)               

                recon_ood_list.append(coordinates_ood[2:, :])
                
            stacked_recon.append(recon_ood_list)
        
    im = plt.imread("./data/intersection_map.png")
    
    plt.figure(figsize=(20,20))
    plt.xlim((200, 1400))
    plt.ylim((1000, 0))
    implot = plt.imshow(im)
    
    if id_plot:
        for coordinates in tqdm(coordinates_list, desc='cooking plot ID'):
            plt.plot(coordinates[:, 0], coordinates[:, 1], c='royalblue', alpha=.1)
    
    colors = ['tab:orange', 'tab:green', 'tab:red']
    
    if z_ood_list is not None:
        for c_i, coordinates_list_ood in enumerate(stacked_recon):
            for coordinates in tqdm(coordinates_list_ood, desc='cooking plot OOD'):
                plt.plot(coordinates[:, 0], coordinates[:, 1], c=colors[c_i], alpha=alpha_ood)
                
    angle2 = 0.2 * np.pi
    rotate2 = np.array([[np.cos(angle2), -np.sin(angle2)],
                       [np.sin(angle2), np.cos(angle2)]])
        
    loc_x_axis_vals = np.transpose(np.vstack([np.arange(-100,100), np.zeros(200)]))
    loc_y_axis_vals = np.transpose(np.vstack([np.zeros(200), np.arange(-100,100)]))
    
    x_axis_coor = np.transpose(rotate2 @ np.transpose(loc_x_axis_vals))*np.array([-1, 1]) + shift
    y_axis_coor = np.transpose(rotate2 @ np.transpose(loc_y_axis_vals))*np.array([-1, 1]) + shift
        
    plt.plot(x_axis_coor[:, 0], x_axis_coor[:, 1], c='black')
    plt.plot(y_axis_coor[:, 0], y_axis_coor[:, 1], c='black')
    
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
                
    if name is not None:
        plt.savefig(name + '.svg')
        plt.savefig(name + '.pdf')
        plt.savefig(name + '.png')

    plt.show()
    
    if z_ood_list is not None and id_plot:
        return coordinates_list, coordinates_list_ood
    
    elif z_ood_list is not None and id_plot is False:
        return coordinates_list_ood
    else:
        return coordinates_list
    
    
def evaluate_reconstruction(loader, autoencoder, plot=True, num_workers=1, device=None):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    ts_list = loader.dataset.get_ts_list()
    
    autoencoder.eval()
    
    names_list = []
    for name, group in loader.dataset.X.groupby(level=loader.dataset.X.index.names[:-1]):
        names_list.append(name)
    
    loss_exp_list = []
    exp_list = []
    counter = 0
    for exp in tqdm(loader.dataset.X.index.unique('ExperimentID'), desc='Exp. evaluated'):
        exp_names = [i for i in names_list if exp in i]

        # loc list of time series for each experiment
        loc_exp_list = ts_list[counter: counter + len(exp_names)]
        counter += len(exp_names)

        # create loader for locked group:
        loc_exp_ts = ConcatDataset(loc_exp_list)
        loader = DataLoader(loc_exp_ts, batch_size=1, shuffle=False, num_workers=num_workers)
        

        loss_list = []
        for x_, y_ in loader:
            x_ = x_.to(device)
            y_ = y_.to(device)

            z = autoencoder.encode(x_.float())
            X_reconstruction = autoencoder.decode(z)

            loss_list.append(F.mse_loss(y_, X_reconstruction).cpu().detach().numpy())
            
        loss_exp_list.append(np.array(loss_list))
        exp_list.append(exp)
        
    if plot:
        plt.figure(figsize=(10,15))
        ax = sns.boxplot(data=loss_exp_list, orient='h', palette="Set2")
        ax.set_yticklabels(exp_list)
        plt.xlabel('MSE')
        plt.ylabel('ExperimentID')
        plt.show()
        
    return loss_exp_list, exp_list