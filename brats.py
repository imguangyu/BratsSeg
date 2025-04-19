import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BraTS(Dataset):
    def __init__(self, data_dir, split='train', modality='flair',transform=None, target_transform=None, slices=[25,126]):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.target_transform = target_transform
        self.slices = slices
        self.num_slice = (self.slices[1] - self.slices[0])

        with np.load(self.data_dir + '.npz',  allow_pickle=True) as loaded_data:
            self.imgs = loaded_data['images'].item()[modality]
            self.labels = loaded_data['labels']
        # self.image_paths = []
        # self.label_paths = []
        # for root, _, files in os.walk(data_dir):
        #     for file in files:
        #         if file.endswith(".npz"):
        #             if "seg" in file:
        #                 self.label_paths.append(os.path.join(root, file))
        #             elif modality in file and 't1ce' not in file:
        #                 self.image_paths.append(os.path.join(root, file))
        
    def __len__(self):
        # return len(self.image_paths) * self.num_slice
        return len(self.imgs)
    
    def _reduce_samples(self, num_samples):
        self.imgs = self.imgs[:num_samples]
        self.labels = self.labels[:num_samples]

    def __getitem__(self, idx):
        # file_idx = idx // self.num_slice
        # slice_idx = idx % self.num_slice

        # image_path = self.image_paths[file_idx]
        # label_path = self.label_paths[file_idx]

        # image = self.load_nifti(image_path)
        # label = self.load_nifti(label_path)

        # image_slice = image[..., slice_idx + self.slices[0]]
        # image_slice = np.expand_dims(image_slice, -1).astype(np.float32)
        # label_slice = label[..., slice_idx + self.slices[0]].astype(np.int32)
        img = self.imgs[idx]
        img = np.expand_dims(img, -1).astype(np.float32)

        label = self.labels[idx].astype(np.int32)


        label[label == 4] = 3

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def load_nifti(self, path):
        # Load nifti file using nibabel
        # return nib.load(path).get_fdata()
        return np.load(path)['data']
    

def fetch_brats(args, root, transforms, modality='flair'):
    logger.info('[LOAD] [BraTS] Fetching dataset!')
    
    # configure arguments for dataset
    dataset_args = {'data_dir': root, 'transform': transforms[0], 'target_transform': transforms[2], 'slices': [75, 76], "modality": modality, "split": "train"}

    # create dataset instance
    raw_train = BraTS(**dataset_args)
    if args.reduce_samples >0 :
        raw_train._reduce_samples(args.reduce_samples)
    elif args.reduce_samples_seg_scale>0:
        raw_train._reduce_samples(int(len(raw_train) * args.reduce_samples_seg_scale))
    raw_train.task = 'seg'
    raw_train.modality = modality
    raw_train.targets = raw_train.labels


    test_args = dataset_args.copy()
    test_args['transform'] = transforms[1]
    test_args['split'] = 'test'

    raw_test = BraTS(**test_args)
    if args.reduce_test_samples >0 and args.reduce_test_samples < len(raw_test):
        raw_test._reduce_samples(args.reduce_samples)
    raw_test.task = 'seg'
    raw_test.modality = modality
    raw_test.targets = raw_test.labels
    
    logger.info('[LOAD] [BraTS] ...fetched dataset!')

    args.in_channels = 1
    args.num_classes = None
    
    return raw_train, raw_test, args

        

    
