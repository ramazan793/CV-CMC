import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision 
import albumentations as A

import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_DIR = '../input/face-points/train'

INPUT_SIZE = 284 # mean across dataset 284, median 170

MyTransform = A.Compose(
    [A.Rotate(limit=45, p = 0.5),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=[0,0.3], p = 0.5), overbrights some classes of images(e.g. old yellow ones). 
    A.OneOf([
        A.RGBShift(r_shift_limit=0, g_shift_limit=[0,100], b_shift_limit=[0,100], p = 0.3),
        A.ToGray(p = 0.5)
    ]),
     A.GaussNoise(p = 0.3),
     A.MotionBlur(p = 0.3)
    ],
    keypoint_params = A.KeypointParams(format='xy')
)

def load_resize(img_path, label, inference = False):
    image = Image.open(img_path).convert('RGB')
    image = np.array(image).astype(np.float32)

    # resize
    if not inference:
        target = np.copy(label)
        target[::2] = label[::2] / image.shape[1] * INPUT_SIZE
        target[1::2] = label[1::2] / image.shape[0] * INPUT_SIZE
        
    x = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    
    if not inference:
        return x, target
    else:
        return x, image.shape

def normalize(img):
    for c in range(3):
        img[:, :, c] -= np.mean(img[:, :, c])
        if np.std(img[:,:,c]) != 0:
            img[:, :, c] /= np.std(img[:, :, c]) 
        else:
            print('Zero std encountered. Check augmentations replay.')
    return img

class MyCustomDataset(Dataset):
    def __init__(self, 
                 mode, 
                 fraction: float = 0.9, 
                 transform = None,
                 store_in_ram = True,
                 data_dir = None,
                 train_gt = None,
                 inference = False,
                 random_state = 1
                ):
        
        self._items = [] 
        self._transform = transform 
        self.store_in_ram = store_in_ram
        self.inference = inference
        
        # train val split
        images_dir = os.listdir(data_dir)
        if not inference:
            np.random.seed(random_state)
            np.random.shuffle(images_dir)
        
        labels = train_gt
        
        partition = int(fraction * len(images_dir))
        if mode == 'train':
            img_names = images_dir[:partition]
        elif mode == 'val':
            img_names = images_dir[partition:]
                
        for img_name in img_names:
            if not inference:
                label = np.array(labels[img_name]).astype(np.float32)
            else:
                label = 0
            
            img_path = os.path.join(data_dir, img_name)
            
            if self.store_in_ram and not inference:
                ## Faster on training, but takes time to preprocess and put into RAM all dataset
                x, target = load_resize(img_path, label)
                
                if np.max(target) > INPUT_SIZE or np.min(target) < 0:
                    print('Label domain mistake. Skipped an image.')
                    continue
            else:
                ## Slower on training. Mainly for inference.
                x = img_path
                target = label
            
            self._items.append((
                x,
                target
            ))
        
    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img, target = self._items[index]
        if not self.store_in_ram or self.inference:
            img, target = load_resize(img, target, inference = self.inference)
        
        if self._transform and not self.inference:
            keypoints = np.zeros((len(target) // 2, 2))
            keypoints[:, 0] = target[::2]
            keypoints[:, 1] = target[1::2]
            transformed = self._transform(image = img, keypoints = keypoints)
            if len(transformed['keypoints']) != 14:
#                 print('Invisible keypoints, skip transform')
                img = np.copy(img).astype(np.float32)
            else:
                img = transformed['image'].astype(np.float32)
                target = np.array(transformed['keypoints'], dtype = np.float32).ravel()
        
        img = normalize(img)

        # to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        return img, target

from torch import nn

class BaseModel(pl.LightningModule):
    # REQUIRED
    def __init__(self, learning_rate):
        super().__init__()
        """ Define computations here. """
        self.lr = learning_rate
        
        self.conv1 = nn.Conv2d(3, 32, 5, padding = 'same') # 62 # 126 # 168 # w/ pad: 170
        self.pool1 = nn.MaxPool2d(2, 2) # 31 # 63 # 84 # w/pad: 85
        self.norm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 'same') # 29 # 61 # 82 # w/ pad: 85
        self.pool2 = nn.MaxPool2d(2, 2) # 14 # 30 # 41 # w/pad: 42
        self.norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 'same') # 12 # 28 # 39 # w/ pad: 42
        self.pool3 = nn.MaxPool2d(2, 2) # 6 # 14 # 19 # w/pad: 21
        self.norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 'same') # 12 # 28 # 39 # w/ pad: 21 35
        self.pool4 = nn.MaxPool2d(2, 2) # 6 # 14 # 19 # w/pad: 21
        self.norm4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding = 'same') # 12 # 28 # 39 # w/ pad: 10 17
        self.pool5 = nn.MaxPool2d(2, 2) # 6 # 14 # 19 # w/pad: 21
        self.norm5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 1024, 3, padding = 'same') # 12 # 28 # 39 # w/ pad: 5 8
        self.pool6 = nn.MaxPool2d(2, 2) # 6 # 14 # 19 # w/pad: 21
        self.norm6 = nn.BatchNorm2d(1024)
        
#         self.global_avg_pool = nn.AvgPool2d(2)
        
        self.fc1 = nn.Linear(1024 * (4)**2, 1024)
        
        self.fc2 = nn.Linear(1024, 28)
        

        self.loss = nn.MSELoss()
        
    # REQUIRED
    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        
        x = self.norm1(self.pool1(F.leaky_relu(self.conv1(x), negative_slope=0.1)))
#         x = F.dropout2d(x, self.p, self.training)
        x = self.norm2(self.pool2(F.leaky_relu(self.conv2(x), negative_slope=0.1)))
#         x = F.dropout2d(x, self.p, self.training)
        x = self.norm3(self.pool3(F.leaky_relu(self.conv3(x), negative_slope=0.1)))
#         x = F.dropout2d(x, self.p, self.training)
        x = self.norm4(self.pool4(F.leaky_relu(self.conv4(x), negative_slope=0.1)))
        x = self.norm5(self.pool5(F.leaky_relu(self.conv5(x), negative_slope=0.1)))
        x = self.norm6(self.pool6(F.leaky_relu(self.conv6(x), negative_slope=0.1)))
        
#         x = self.global_avg_pool(x)
        
        x = torch.flatten(x, 1)
#         print(x.shape)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1, self.training)
        
        x = self.fc2(x) 
        
        return x
    
    # REQUIRED
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        self.training = True
        
        x, y = batch

        y_pred = self(x)
        loss = self.loss(y_pred, y)
        
        eps = 5
        acc = torch.sum(torch.abs(y_pred.detach() - y) < eps) / y.shape[0] / y.shape[1]
        
        system_loss = torch.mean(((y_pred.detach() - y) / INPUT_SIZE * 100)**2)
        
        return {'loss': loss, 'acc': acc, 'train_sys' : system_loss}
    
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = 5e-4)
#         optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode='min', 
                                                                  factor=0.5, 
                                                                  patience=10, 
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
        self.training = False
        x, y = batch
        y_pred = self(x)

        loss = self.loss(y_pred, y)
    
        eps = 5
        acc = torch.sum(torch.abs(y_pred - y) < eps) / y.shape[0] / y.shape[1]
        
        system_loss = torch.mean(((y_pred - y) / INPUT_SIZE * 100)**2)
        
        return {'val_loss': loss, 'val_acc': acc, 'val_sys' : system_loss}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_sys = torch.stack([x['train_sys'] for x in outputs]).mean()
        
        print(f"| Train_acc: {avg_acc:.2f}, Train_loss: {avg_loss:.2f}, Train_sys: {avg_sys:.2f}" )
        
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', avg_acc, prog_bar=True, on_epoch=True, on_step=False)
     
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_sys = torch.stack([x['val_sys'] for x in outputs]).mean()
        
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_acc: {avg_acc:.2f}, Val_loss: {avg_loss:.2f}, Val_sys: {avg_sys:.2f}", end= " ")
        
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', avg_acc, prog_bar=True, on_epoch=True, on_step=False)


def train_detector(train_gt, train_img_dir, fast_train = True):
    if not fast_train:
        BATCH_SIZE = 128
        NUM_EPOCHS = 200
    else:
        BATCH_SIZE = 1
        NUM_EPOCHS = 1
        
    ## Init train and val datasets
    ds_train = MyCustomDataset(mode = "train", data_dir = train_img_dir, train_gt = train_gt, transform = MyTransform, store_in_ram = False if fast_train else True)
    ds_val = MyCustomDataset(mode = "val", data_dir = train_img_dir, train_gt = train_gt, transform = MyTransform, store_in_ram = False if fast_train else True)

    ## Init train and val dataloaders
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    LEARNING_RATE = 1e-3
    
    trainer = pl.Trainer(
        max_epochs = NUM_EPOCHS,
        max_steps = 1 if fast_train else -1,
        gpus = 1 if torch.cuda.is_available() else 0,
        callbacks = None,
        logger = False,
        checkpoint_callback = False,
        num_sanity_val_steps=0
    )
    
    model = BaseModel(learning_rate = LEARNING_RATE)

    trainer.fit(model, dl_train, dl_val)
    
    return model

def detect(model_filename, test_img_dir):
    model = BaseModel.load_from_checkpoint(model_filename, learning_rate = 1e-3)
    model.eval()
    model.training = False
    model.inference = True
    model.to(device)
    
    ans = dict()
    
    for img_name in tqdm(os.listdir(test_img_dir)):
        img_path = os.path.join(test_img_dir, img_name)
        img, img_shape = load_resize(img_path, [], inference = True)
        img = normalize(img)
        
        inp = torch.from_numpy(img.transpose(2, 0, 1))[None, :].to(device)

        res = model(inp).detach()

        res[::2] = res[::2] / INPUT_SIZE * img_shape[1]
        res[1::2] = res[1::2] / INPUT_SIZE * img_shape[0]

        res = np.clip(res.cpu().numpy(), 0, max(img_shape[0],img_shape[1])).astype(np.int32)
        
        ans[img_name] = res.ravel()
    
    return ans