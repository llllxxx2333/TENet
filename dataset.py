import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
def cv_random_flip(img, label, depth,thermal):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth,thermal


def randomCrop(image, label, depth,thermal ):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region),thermal.crop(random_region)


def randomRotation(image, label, depth, thermal):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        thermal = thermal.rotate(random_angle, mode)
    return image, label, depth, thermal


def colorEnhance(image):
    #亮度
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    #对比度
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    #色度
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    #锐度
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
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


# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root, ti_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) ]
        # print(self.images)
        self.gts = [gt_root + f for f in os.listdir(gt_root) ]
        self.depths = [depth_root + f for f in os.listdir(depth_root)]
        self.tis = [ti_root + f for f in os.listdir(ti_root) ]


        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.tis = sorted(self.tis)


        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        ti = self.rgb_loader(self.tis[index])
        depth = self.binary_loader(self.depths[index])


        image, gt, depth, ti = cv_random_flip(image, gt, depth,ti)
        image, gt, depth, ti = randomCrop(image, gt, depth,ti)
        image, gt, depth, ti = randomRotation(image, gt, depth, ti)
        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        ti = self.tis_transform(ti)

        return image, gt,depth, ti

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.tis)
        images = []
        gts = []
        tis = []
        depths = []
        edges = []

        for img_path, gt_path,depth_path, ti_path in zip(self.images, self.gts, self.depths,self.tis):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            ti = Image.open(ti_path)

            if img.size == gt.size and gt.size == ti.size:
                images.append(img_path)
                gts.append(gt_path)
                tis.append(ti_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths
        self.tis = tis


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt,depth, ti):
        assert img.size == gt.size and gt.size == ti.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                depth.resize((w, h), Image.NEAREST),ti.resize((w, h),Image.NEAREST)
        else:
            return img, gt, depth,ti

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, depth_root,ti_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root,ti_root, trainsize)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    # print(len(data_loader))
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root,ti_root,testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) ]
        self.gts = [gt_root + f for f in os.listdir(gt_root) ]
        self.depths = [depth_root + f for f in os.listdir(depth_root) ]
        self.tis = [ti_root + f for f in os.listdir(ti_root) ]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.tis = sorted(self.tis)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth)
        depth = depth.unsqueeze(0)
        ti = self.rgb_loader(self.tis[self.index])
        ti = self.tis_transform(ti).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth,ti,name

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

