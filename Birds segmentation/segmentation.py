from torch.utils.data import Dataset, DataLoader
import torchvision.models
# from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch

import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# import matplotlib.pyplot as plt

import os

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class, pretrained = True):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2) # convrelu(3, 64, 7, stride = 2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
    
    
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +\
                                                 target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

class MyModel(pl.LightningModule):
    # REQUIRED
    def __init__(self, num_classes = 2, lr = 1e-4, pretrained = True):
        super().__init__()
        """ Define computations here. """
        
        self.lr = lr
        self.model = ResNetUNet(num_classes, pretrained)
        
        # freeze backbone layers
        for l in self.model.base_layers[:-3]:
            for param in l.parameters():
                param.requires_grad = False
        
        
        self.bce_weight = 0.9
    
    # REQUIRED
    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.model(x)
        return x
    
    
    # REQUIRED
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch
        
        y = y[:, None, :, :].float() / 255
        
        y_logit = self(x)
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        return {'loss': loss}
    
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode='min', 
                                                                  factor=0.1, 
                                                                  patience=3, 
                                                                  verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        } 
        
        return [optimizer], [lr_dict]
    
    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch
        
        y = y[:, None, :, :].float() / 255
        
        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        return {'val_loss': loss, 'logs':{'dice':dice, 'bce': bce}}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        print(f"| Train_loss: {avg_loss:.3f}" )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()
        
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.3f}, Val_dice: {avg_dice:.3f}, Val_bce: {avg_bce:.3f}", end= " ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        
        
INPUT_SIZE = 256


def secure2rgb(img):
    if len(img.shape) != 3:
        img = np.dstack([img, img, img])
    return img    

class BirdsDataset(Dataset):
    """
    :param root_folders: список путей до папок с данными
    """
    def __init__(self, folder, test = False) -> None:
        self._items = []
        self.test = test
        
        images_folder = os.path.join(folder, 'images')
        gt_folder = os.path.join(folder, 'gt')
        
        for folder in os.listdir(images_folder):
            for fname in os.listdir(os.path.join(images_folder, folder)):
                img_path = os.path.join(images_folder, folder, fname)
                gt_path = os.path.join(gt_folder, folder, fname)
                gt_path = gt_path[:-3] + 'png'
                self._items.append((img_path, gt_path))

        if not self.test:
            self.transform = A.Compose([
                A.Rotate(15),
                A.RGBShift(10, 10, 10),
                A.GaussNoise(),
                A.Resize(INPUT_SIZE, INPUT_SIZE),
                A.Normalize(),
                ToTensorV2()
            ]) 
        else:
            self.transform = A.Compose([
                A.Resize(INPUT_SIZE, INPUT_SIZE),
                A.Normalize(),
                ToTensorV2()
            ])
            
    def __getitem__(self, index):
        img_path, gt_path = self._items[index]
        img = np.asarray(Image.open(img_path), dtype = np.uint8)
        mask = np.asarray(Image.open(gt_path), dtype = np.uint8)
        
        # convert gray to rgb
        img = secure2rgb(img)
        
        tr = self.transform(image = img, mask = mask)
        transformed_img, gt = tr['image'], tr['mask']
        if len(gt.shape) != 2:
            gt = gt[:, :, 0]
        
        return transformed_img, gt
    
    def __len__(self):
        return self._items.__len__()
    
def train_segmentation_model(train_data_path):
    BATCH_SIZE = 32

    train_ds = BirdsDataset(train_data_path)
    train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)

    model = MyModel(num_classes = 1, lr = 1e-4).to(device)

    trainer = pl.Trainer(
        max_epochs=11,
        gpus=1,
        log_every_n_steps=5,
        callbacks = None,
        logger = False,
        checkpoint_callback = False
    )
    
    trainer.fit(model, train_dl)
    
    for key, param in model.model.state_dict().items():
        param[param.abs() < 1e-36] = 0
    
    torch.save(model.state_dict(), 'segmentation_model.pth')
    
    return model

def get_model():
    with torch.no_grad():
        model = MyModel(num_classes = 1, lr = 1e-4, pretrained = False).to(device)
        model.eval()
        return model

def predict(model, img_path):
    with torch.no_grad():
        transform = A.Compose([
                    A.Resize(INPUT_SIZE, INPUT_SIZE),
                    A.Normalize(),
                    ToTensorV2()
                ])
        img = np.asarray(Image.open(img_path), dtype = np.uint8)
        size = img.shape[:2]
        img = secure2rgb(img)
        img = transform(image = img)['image']
        img = img[None, :, :, :].to(device)

        segm = model(img).detach().cpu().numpy()[0, 0]

        segm = A.Resize(*size)(image = segm)['image']

        return segm