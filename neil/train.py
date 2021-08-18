"""Training function."""
import os

from glob import glob
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tqdm import tqdm

from utils import extract_audio, get_fmri, predict_fmri_fast, vectorized_correlation

_RANDOM_STATE = 42
_BATCH_SIZE = 64

def _load_data(sub, ROI, fmri_dir='../../Algonauts2021_devkit/participants_data_v2021', batch_size=128, use_gpu=True):
    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"

    fmri_dir = os.path.join(fmri_dir, track)
    sub_fmri_dir = os.path.join(fmri_dir, sub)
    # results_dir = os.path.join('../results/', f'{image_model.__class__.__name__}', track, sub)

    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir, ROI)

    return fmri_train_all

class VoxelDataset(Dataset):
    def __init__(self, features, voxel_maps=None, transform=None):
        self.features = features

        self.voxel_maps = voxel_maps

        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        voxel_map = torch.tensor(self.voxel_maps[idx].astype(np.float32))

        if self.transform:
            feat = self.transform(feat)

        return feat, voxel_map

class MultiVoxelAutoEncoder(pl.LightningModule):
    def __init__(self, in_features=1668, out_features=368):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, out_features)
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(out_features, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, in_features)
        # )

    def forward(self, x):
        x = self.encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)

        y_hat = self.encoder(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'loss': {'train': loss}}
        return {"loss": loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # z = self.encoder(x)
        # x_hat = self.decoder(x)
        # loss = F.mse_loss(x_hat, x)

        y_hat = self.encoder(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        tensorboard_logs = {'loss': {'val': loss}}
        return {"val_loss": loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    device = torch.device('cuda')
    seed_everything(_RANDOM_STATE)

    video_paths = sorted(glob('../../Algonauts2021_devkit/AlgonautsVideos268_All_30fpsmax/*.mp4'))
    audio_paths = sorted(glob('../../Algonauts2021_devkit/AlgonautsVideos268_All_30fpsmax/*.wav'))
    sub_folders = sorted(os.listdir('../../Algonauts2021_devkit/participants_data_v2021/mini_track/'))
    mini_track_ROIS = sorted(list(map(lambda x: Path(x).stem, os.listdir('../../Algonauts2021_devkit/participants_data_v2021/mini_track/sub01/'))))

    if audio_paths is None:
        for video_path in video_paths:
            extract_audio(video_path)

    print(f'Found {len(video_paths)} videos, {len(audio_paths)} audios')

    all_embeddings = np.load('densenet169_concat_all_blocks.npy')
    print(f'Embedding dimensions: {(all_embeddings.shape)}')
    train_embeddings, test_embeddings = np.split(all_embeddings, [1000], axis=0)

    sub = 'sub04'
    ROI = 'EBA'

    batch_size = 32

    train_voxels = _load_data(sub, ROI)

    train_embeddings, val_embeddings, train_voxels, val_voxels = train_test_split(train_embeddings, train_voxels, train_size=0.9)

    train_dataset = VoxelDataset(train_embeddings, train_voxels)
    val_dataset = VoxelDataset(val_embeddings, val_voxels)

    train_loader = DataLoader(train_dataset, batch_size=_BATCH_SIZE)
    val_loader = DataLoader(val_dataset)

    print(train_embeddings.shape, train_voxels.shape, val_embeddings.shape, val_voxels.shape)

    mvae = MultiVoxelAutoEncoder(in_features=train_embeddings.shape[1], out_features=train_voxels.shape[1])

    # early_stopping = EarlyStopping('val_loss', patience=5)
    trainer = pl.Trainer(
        max_epochs=30,
        gpus=2,
        # callbacks=[early_stopping],
        plugins=DDPPlugin(find_unused_parameters=False)
    )
    trainer.fit(mvae, train_loader)

    mvae.eval()
    for param in mvae.parameters():
        param.requires_grad = False

    y_pred = mvae(torch.tensor(val_embeddings)).detach().numpy()
    assert(y_pred.shape == val_voxels.shape)

    r = vectorized_correlation(val_voxels, y_pred)
    print(f'Validation correlation for {sub}/ROI={ROI}: {r.mean():.4f}')

    # print('-' * 20)
    # print('Comparison to the PCA/OLS technique:')

    # pred_fmri = np.zeros_like(val_voxels)
    # for i in tqdm(range(0, train_voxels.shape[1] - batch_size, batch_size), ascii=True):
    #     pred_fmri[:, i:i + batch_size] = predict_fmri_fast(train_embeddings, test_embeddings, train_voxels[:, i:i + batch_size], use_gpu=True)

    # pred_fmri[:, i:] = predict_fmri_fast(train_embeddings, test_embeddings, train_voxels[:, i:], use_gpu=True)

    # score = vectorized_correlation(val_voxels, y_pred)
    # mean_score = round(score.mean(), 3)
    # print("----------------------------------------------------------------------------")
    # print("Mean correlation for ROI : ", ROI, "in ", sub, " is :", mean_score)
