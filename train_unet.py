import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import torch.optim as optim
from torch.backends import cudnn
import tqdm

# define model
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32 * factor, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class LungSegDataset(Dataset):
    def __init__(self, images, masks):
        super(LungSegDataset, self).__init__()
        self.images = images
        self.masks = masks
    def __getitem__(self, index):
        image = self.images[index]
        image = np.transpose(image, [2, 0, 1])
#         image = (image - image.min()) / (image.max() - image.min())
        mask = self.masks[index].astype(int)
        return torch.from_numpy(image), torch.from_numpy(mask)
        
    def __len__(self):
        return len(self.images)

class COVIDLungSegDataset(Dataset):
    def __init__(self, image_list):
        super(COVIDLungSegDataset, self).__init__()
        self.images = image_list
        
    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(image_path, 0).astype(np.float32)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=2)
        image /= 255.0
        image = image.transpose([2, 0, 1])
        return torch.from_numpy(image), image_path
        
    def __len__(self):
        return len(self.images)

def test_covid_seg():
    import sys
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    batch_size = 32
    seed = 42
    model_path = './exps/unet/models/unet_39.pth'
    torch.manual_seed(seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(seed)
    model = UNet(1, 2)
    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    model = model.cuda()
    image_file = sys.argv[1]
    image_list = []
    output_dir = './data_segmentation_test'
    with open(image_file, 'r') as reader:
        for line in reader:
            line = line.strip()
            if line:
                image_list.append(line)
    test_dataset = COVIDLungSegDataset(image_list)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        model.eval()
        for sample in tqdm.tqdm(test_dataloader):
            images, paths = sample 
            images = images.cuda()
            masks = model(images)
            masks = F.softmax(masks, dim=1)
            masks = masks.cpu().numpy()
            masks = masks.transpose([0, 2, 3, 1])
            masks *= 255.0
            masks = np.round(masks).astype(np.uint8)
            masks = masks[:, :, :, 1]
            for index, mask in enumerate(masks):
                path = paths[index]
                mask_path = os.path.join(output_dir, '/'.join(path.split('/')[1:]))
                if not os.path.isdir(os.path.dirname(mask_path)):
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                cv2.imwrite(mask_path, mask)
def train():
    import os
    import logging
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    epoches = 40
    batch_size = 16
    seed = 42
    log_step = 5
    log_output_dir = './exps/unet/logs'
    model_output_dir = './exps/unet/models'
    if not os.path.isdir(log_output_dir):
        os.makedirs(log_output_dir, exist_ok=True)
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)

    torch.manual_seed(seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(seed)
    x_train = np.load('./lung_segmentation_data/x_train.npy')
    y_train = np.load('./lung_segmentation_data/y_train.npy')
    x_val = np.load('./lung_segmentation_data/x_val.npy')
    y_val = np.load('./lung_segmentation_data/y_val.npy')
    x_train = np.concatenate([x_train, x_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)
    model = UNet(1, 2)
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    train_dataset = LungSegDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=False)
    # val_dataset = LungSegDataset(x_val, y_val)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    for epoch in range(epoches):
        model.train()
        epoch_loss = 0.0
        print('************************************')
        for step, sample in enumerate(train_dataloader):
            optimizer.zero_grad
            images, true_masks = sample
            images = images.cuda()
            true_masks = true_masks.cuda()
            pred_masks = model(images)
            pred_masks = pred_masks.transpose(1, 2).transpose(2, 3).contiguous()
            pred_masks = pred_masks.reshape(-1, 2)
            true_masks = true_masks.reshape(-1)
            loss = criterion(pred_masks, true_masks)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.25)
            optimizer.step()
            _loss = loss.item()
            epoch_loss += _loss
            if step % log_step == 0 or step == len(train_dataloader) - 1:
                logging.info('epoch {} step {} loss is {}/{}'.format(epoch, step, _loss, epoch_loss / (step + 1)))
        model_path = os.path.join(model_output_dir, 'unet_{}.pth'.format(epoch))
        model.eval()
        info_to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            'epoch':epoch
        }
        torch.save(info_to_save, model_path)

         
if __name__ == '__main__':
    train()
#     test_covid_seg()
    


