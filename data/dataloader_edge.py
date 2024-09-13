import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import random

# several data augumentation strategies
def cv_random_flip(img, gt, edge):
    # print("cv_random_flip")
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # fix = fix.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, gt, edge


def randomCrop(image, gt, edge):
    # print("randomCrop")
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    # return image.crop(random_region), fix.crop(random_region), gt.crop(random_region)
    return image.crop(random_region), gt.crop(random_region), edge.crop(random_region)

def randomRotation(image, gt, edge):
    # print("randomRotation")
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        # fix = fix.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, gt, edge


def colorEnhance(image):
    # print("colorEnhance")
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image



def randomPeper(img):
    # print("randomPeper")
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.kernel = np.ones((3, 3), np.uint8)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = self.edge_loader(self.edges[index])

        image, gt, edge = cv_random_flip(image, gt, edge)
        image, gt, edge = randomCrop(image, gt, edge)
        image, gt, edge = randomRotation(image, gt, edge)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        edge = randomPeper(edge)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        edge = self.edge_transform(edge)

        edge_th = self.Threshold_process(edge)
        name = self.images[index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, gt, name, edge_th

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        edges = []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and img.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def edge_loader(self, path):
        edge = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge)
        return edge

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class test_dataset(data.Dataset):
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, edge_root, trainsize)
    dataset_num = dataset.__len__()

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)


    return data_loader, dataset_num

def get_testloader(image_root, gt_root, batchsize, testsize, shuffle=False, num_workers=0, pin_memory=True):
    testdataset = test_dataset(image_root, gt_root, testsize)
    dataset_num = testdataset.__len__()
    data_testloader = data.DataLoader(dataset=testdataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_testloader, dataset_num
