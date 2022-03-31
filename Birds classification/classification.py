import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
# from torchsummary import summary
import torchvision 
import albumentations as A

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os

from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

INPUT_SIZE = (288, 288)
BATCH_SIZE = 32
NUM_CLASSES = 50


def identify_transform(image):
    return { 'image' : np.copy(image) }

expand_transforms = [
    identify_transform,
    A.Rotate(limit = 90, always_apply = True),
    A.HueSaturationValue(40, 10, 10, always_apply = True),
#     A.ToGray(always_apply = True),
    A.RandomBrightnessContrast(always_apply = True)
]


def normalize(image):
    x = np.copy(image).astype(np.float32)
    for c in range(3):
        x[:, :, c] -= np.mean(x[:, :, c])
        if np.std(x[:,:,c]) != 0:
            x[:, :, c] /= np.std(x[:, :, c]) 
        else:
            print('Zero std encountered', )
    return x

def load(img_path):
    image = Image.open(img_path).convert('RGB')
    image = np.array(image).astype(np.uint8)
    return image

def resize(image, size):
    image = cv2.resize(image, size[::-1])
    return image

def crop(image, size):
    try:
        image = A.CenterCrop(*size)(image = image)['image']
    except Exception: # crop size larger than image size
        image = resize(image, size)
    return image

def img2tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1))

class MyCustomDataset(Dataset):
    def __init__(self, 
                 mode, 
                 data_dir : str, 
                 test_size: float = 0.2, 
                 transforms = [identify_transform],
                 store_in_ram = True,
                 labels = None,
                 random_state = 42
                ):
        
        self._items = []         
        self._transforms = transforms
        self._ram = store_in_ram
        
        images_dir = os.listdir(data_dir)
        labels_raw = list(labels.values())
        X_train, X_test = train_test_split(images_dir, test_size = test_size, random_state = random_state, shuffle = True, stratify = labels_raw)
        
        if mode == 'train':
            img_names = X_train
        elif mode == 'val':
            img_names = X_test

        for img_name in img_names:
            image_path = os.path.join(data_dir, img_name)
            label = labels[img_name]      
            
            # Dataset expanding
            if store_in_ram:
                src_image = load(image_path)
                images = []
                
                for transform in transforms:
                    transformed = transform(image = src_image)['image']
                    transformed = crop(transformed, INPUT_SIZE)
                    transformed = normalize(transformed)
                    transformed = img2tensor(transformed)
                    images.append(transformed)
            
                for img in images:
                    self._items.append((
                        img,
                        label
                    ))
            else:
                for i in range(len(transforms)):
                    img_transform = i
                    self._items.append((
                        image_path,
                        img_transform,
                        label
                    ))
                
        np.random.shuffle(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        if self._ram:
            return self._items.__getitem__(index)
        else:
            img_path, img_transform, label = self._items.__getitem__(index)
            image = load(img_path)
            image = self._transforms[img_transform](image = image)['image']
            image = crop(image, INPUT_SIZE)
            image = normalize(image)
            image = img2tensor(image)
            
            return image, label
    
def mobilenet_v2_fe(trainable_params, fast_train):
    model = torchvision.models.mobilenet_v2(pretrained = not fast_train)
    feature_extractor = model.features

    for param in list(feature_extractor.parameters())[:-trainable_params]:
        param.requires_grad = False
    
    return feature_extractor, model.last_channel

def resnext50_bf(trainable_params, fast_train): # 3k
    model = torchvision.models.resnext50_32x4d(pretrained = not fast_train)
    
    params_count = len(list(model.parameters()))
    
    params = list(model.parameters())[:-2] # drop last fc layer, we will change it
    
    for param in params[:-trainable_params]:
        param.requires_grad = False
    
    return model, 2048

def densenet201_fe(trainable_params, fast_train): # 2 + 3k
    model = torchvision.models.densenet201(pretrained = not fast_train)
    feature_extractor = model.features

    for param in list(feature_extractor.parameters())[:-trainable_params]:
        param.requires_grad = False
    
    return feature_extractor, 64

class BirdsClassifier(pl.LightningModule):
    # REQUIRED
    def __init__(self, lr = 3e-4, fast_train = False):
        super().__init__()
        ### mobilenetv2 setup
#         self.backbone, last_channel = mobilenet_v2_fe(trainable_params = 3, fast_train = fast_train)
        
#         self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(last_channel, NUM_CLASSES),
#         )
        
        ### resnext-50 setup
        self.backbone, input1d = resnext50_bf(trainable_params = 12, fast_train = fast_train)  # 3
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4, inplace = True),
#             nn.Linear(input1d, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2, inplace = True),
            nn.Linear(input1d, NUM_CLASSES),
        )
        
        ### efficentnet_b2 setup
#         self.backbone, last_channel = densenet201_fe(trainable_params = 3, fast_train = fast_train)
        
#         self.relu = nn.ReLU()
#         self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
        
#         self.classifier = nn.Sequential(
# #             nn.Dropout(0.2, inplace=True),
#             nn.Linear(last_channel, NUM_CLASSES),
#         )
        
        self.loss = nn.CrossEntropyLoss()
        
        self.lr = lr
        
    # REQUIRED
    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.backbone(x)
        
#         x = self.ReLU(x, inplace = True)
#         x = self.global_pooling(x)
#         x = self.flatten(x)
        
#         x = self.classifier(x)
        return x
    
    # REQUIRED
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        self.train()
        self.backbone.train()
        
        x, y = batch

        y_pred = self(x)
        
        loss = self.loss(y_pred, y)
        
        eps = 5
        acc = torch.sum(y_pred.detach().argmax(dim = 1) == y) / y.shape[0]
        
        return {'loss': loss, 'acc': acc}
    
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = 5e-4)
        
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                                                   mode='min', 
#                                                                   factor=0.5, 
#                                                                   patience=10, 
#                                                                   verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                            T_0=5, 
                                                                            T_mult=1, 
                                                                            eta_min=4e-3, 
                                                                            last_epoch=-1,
                                                                            verbose=True)
        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss"
        } 
        
        return [optimizer], [lr_dict]
    
    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        self.eval()
        self.backbone.eval()
        
        x, y = batch
        y_pred = self(x)
        
        loss = self.loss(y_pred, y)
    
        acc = torch.sum(y_pred.argmax(dim = 1) == y) / y.shape[0]
        
        return {'val_loss': loss, 'val_acc': acc}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        
        print(f"| Train_acc: {avg_acc:.2f}, Train_loss: {avg_loss:.2f}" )
        
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', avg_acc, prog_bar=True, on_epoch=True, on_step=False)
     
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_acc: {avg_acc:.2f}, Val_loss: {avg_loss:.2f}", end= " ")
        
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', avg_acc, prog_bar=True, on_epoch=True, on_step=False)
        
def train_classifier(train_gt, train_img_dir, fast_train = True):
    if not fast_train:
        BATCH_SIZE = 128
        NUM_EPOCHS = 200
    else:
        BATCH_SIZE = 1
        NUM_EPOCHS = 1
        
    ## Init train and val datasets
    ds_train = MyCustomDataset(mode = "train", data_dir = train_img_dir, transforms = expand_transforms, store_in_ram = False, labels = train_gt)
    ds_val = MyCustomDataset(mode = "val", data_dir = train_img_dir, store_in_ram = False, labels = train_gt)

    ## Init train and val dataloaders
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    LEARNING_RATE = 1e-5
    
    trainer = pl.Trainer(
        max_epochs = NUM_EPOCHS,
        max_steps = 1 if fast_train else -1,
        gpus = 1 if torch.cuda.is_available() else 0,
        callbacks = None,
        logger = False,
        checkpoint_callback = False,
        num_sanity_val_steps=0
    )
    
    model = BirdsClassifier(LEARNING_RATE, fast_train = fast_train)

    trainer.fit(model, dl_train, dl_val)
    
    return model

def classify(model_filename, test_img_dir):
    model = BirdsClassifier.load_from_checkpoint(model_filename, learning_rate = 1e-3, fast_train = True)
    model.eval()
    model.to(device)
    
    ans = dict()
    
    for img_name in tqdm(os.listdir(test_img_dir)):
        img_path = os.path.join(test_img_dir, img_name)
        img = load(img_path)
        img = crop(img, INPUT_SIZE)
        img = normalize(img)
        img = img2tensor(img)
        
        inp = img[None, :].to(device)

        res = model(inp).detach()

        res = res.argmax().cpu().numpy()
        ans[img_name] = res
    
    return ans