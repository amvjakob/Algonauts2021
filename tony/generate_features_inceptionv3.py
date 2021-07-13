
import glob
from inceptionv3 import *
import numpy as np
import urllib
import torch
import cv2
import argparse
import time
import random
from tqdm import tqdm
from torchvision import transforms as trn
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable as V
from sklearn.decomposition import PCA, IncrementalPCA
from decord import VideoReader
from decord import cpu

seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

def sample_video_from_mp4(file, num_frames=16):
    """This function takes a mp4 video file as input and returns
    a list of uniformly sampled frames (PIL Image).
    Parameters
    ----------
    file : str
        path to mp4 video file
    num_frames : int
        how many frames to select using uniform frame sampling.
    Returns
    -------
    images: list of PIL Images
    num_frames: int
        number of frames extracted
    """
    images = list()
    vr = VideoReader(file, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames-1, num_frames, dtype=np.int)
    for seg_ind in indices:
        images.append(Image.fromarray(vr[seg_ind].asnumpy()))
    return images, num_frames


def get_activations_and_save(model, video_list, activations_dir):
    """This function generates features and save them in a specified directory.
    Parameters
    ----------
    model :
        pytorch model : alexnet.
    video_list : list
        the list contains path to all videos.
    activations_dir : str
        save path for extracted features.
    """

    resize_normalize = trn.Compose([
        trn.Resize(299),
        trn.CenterCrop(299),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    for video_file in tqdm(video_list):
        vid, num_frames = sample_video_from_mp4(video_file)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = []
        for frame, img in enumerate(vid):
            input_img = V(resize_normalize(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.cuda()
            x = model.forward(input_img)
            for i, feat in enumerate(x):
                if frame == 0:
                    activations.append(feat.data.cpu().numpy().ravel())
                else:
                    activations[i] = activations[i] + \
                        feat.data.cpu().numpy().ravel()
        for layer in range(len(activations)):
            save_path = os.path.join(
                activations_dir, video_file_name+"_"+"layer" + "_" + str(layer+1) + ".npy")
            avg_layer_activation = activations[layer]/float(num_frames)
            np.save(save_path, avg_layer_activation)


def do_PCA_and_save(activations_dir, save_dir, n_components=100):
    """This function preprocesses Neural Network features using PCA and save the results
    in a specified directory
.

    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.

    """

    layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4',
              'layer_5', 'layer_6', 'layer_7', 'layer_8']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir + '/*'+layer+'.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list), feature_dim.shape[0]))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i, :] = temp
        x_train = x[:1000, :]
        x_test = x[1000:, :]

        start_time = time.time()
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components, random_state=seed)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)
        train_save_path = os.path.join(save_dir, "train_"+layer)
        test_save_path = os.path.join(save_dir, "test_"+layer)
        np.save(train_save_path, x_train)
        np.save(test_save_path, x_test)


def main():
    parser = argparse.ArgumentParser(
        description='Feature Extraction from InceptionV3 and preprocessing using PCA')
    parser.add_argument('-vdir', '--video_data_dir', help='video data directory',
                        default='./AlgonautsVideos268_All_30fpsmax/', type=str)
    parser.add_argument('-sdir', '--save_dir',
                        help='saves processed features', default='./inceptionv3', type=str)
    args = vars(parser.parse_args())

    save_dir = args['save_dir']
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    video_dir = args['video_data_dir']
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    model = inceptionv3()

    # get and save activations
    print("------------------- Saving activations ----------------------------")
    activations_dir = os.path.join(save_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    get_activations_and_save(model, video_list, activations_dir)

    # preprocessing using PCA and save
    print("---------------------- Performing PCA -----------------------------")
    pca_dir = os.path.join(save_dir, 'pca_100')
    do_PCA_and_save(activations_dir, pca_dir, n_components=100)


if __name__ == "__main__":
    main()
