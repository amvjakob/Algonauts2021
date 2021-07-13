import nibabel as nib
import pickle
import numpy as np

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
        #print(p)
        #ret_di = pickle.load(f)
    return ret_di

def saveasnii(brain_mask,nii_save_path,nii_data):
    img = nib.load(brain_mask)
    print(img.shape)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)

def get_region_mask(data_dir, region):
    region = load_dict(data_dir + region + ".pkl")
    wb = load_dict(data_dir + "/WB.pkl")
    
    mask_idx = np.where(wb['voxel_mask'] == 1)
    minimask = [
        np.where(wb['train'][0, 0, :] == el)[0][0] 
        for el in region['train'][0, 0, :]
    ]
    minimask = [mask_idx[i][minimask] for i in range(len(mask_idx))]
    
    mask = np.zeros((78,93,71))
    mask[tuple(minimask)] = 1

    return mask
