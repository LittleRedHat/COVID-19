import os
import sys

import torch
from torch.backends import cudnn
import numpy as np
import cv2
from model import COVIDNet


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    seed = 42
    n_feats = 2048
    image_name = '东院独立验证集/12061587/ser003img00029.jpg'
#     image_path = './data_selected/{}'.format(image_name)
    mask_path = './data_segmentation_test/{}'.format(image_name)
    image_path = './data_test/{}'.format(image_name)
    output_name = '_'.join(os.path.splitext(image_name)[0].split('/'))
    grad_cam_output = output_name + '_cam.jpg'
    best_epoch = 12
    size = (256, 256)
    torch.manual_seed(seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(seed)
    
    model = COVIDNet(2, n_feats=n_feats)
    model_path = './exps/covidnet/1/models/covidnet_{}.pth'.format(best_epoch)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])
    print('loading model')
    model.cuda()
    model.eval()
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('{}_origin.jpg'.format(output_name), image)
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_CUBIC)
    image[mask < 200, :] = 0
    cv2.imwrite('{}_mask.jpg'.format(output_name), image)
    image = image.astype(np.float32)
    
    image /= 255.0
    image = image.transpose([2, 0, 1])
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).unsqueeze(0)
    image = image.cuda()
    logits, features = model(image, return_feats=True)
    cam = grad_cam(model, features, logits, index=1)
    show_cam_on_image(image_path, size, cam, grad_cam_output)
    
def grad_cam(model, features, logits, index=None, use_cuda=True):
    class_num = logits.size(-1)
    if index == None:
        index = torch.argmax(logits).item()

    one_hot = np.zeros((1, class_num),dtype=np.float32)
    one_hot[:,index] = 1

    one_hot = torch.tensor(one_hot, dtype=torch.float, requires_grad=True)
    
    if use_cuda:
        one_hot = one_hot.cuda()

    one_hot = torch.sum(one_hot * logits)
    model.zero_grad()
    
    ## 1 * c * w * h
    target = features[0].cpu().data.numpy()
    one_hot.backward(retain_graph=True)
    gradient = model.gradients[0]
    gradient = gradient.cpu().data.numpy()    
    weights = np.mean(gradient, axis=(2, 3))[0, :]
    cam = np.zeros(target.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * target[i, :, :]
    return cam
    
def show_cam_on_image(image_path, image_size, cam, output_name):
    img = cv2.imread(image_path)
    img = np.float32(cv2.resize(img, image_size)) / 255

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, image_size)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    cv2.imwrite(output_name, np.uint8(cam * 255))
    

if __name__ == '__main__':
    main()