import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from torchvision import datasets, transforms
from torchvision.transforms import Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Calculate mean and standard deviation for training dataset
train_data = datasets.CIFAR10('./data', download=True, train=True)

# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
print(x.shape)
# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x, axis=(0, 1))/255
train_std = np.std(x, axis=(0, 1))/255
# print the mean and std
print(train_mean, train_std)

train_transforms = A.Compose([
    A.Normalize(mean=train_mean, std=train_std),
    A.PadIfNeeded(min_height=36, min_width=36, p=1),
    A.RandomCrop(32, 32, p=1),
    A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, 
                    min_width=16, fill_value=train_mean, mask_fill_value = None),
    ToTensorV2(),
])
    
test_transforms = A.Compose([
    A.Normalize(mean=train_mean, std=train_std),
    ToTensorV2(),
])


def get_incorrect_preds(model, test_dataloader):
  incorrect_examples = []
  pred_wrong = []
  true_wrong = []

  model.eval()
  for data,target in test_dataloader:
    data , target = data.cuda(), target.cuda()
    output = model(data)
    _, preds = torch.max(output,1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    preds = np.reshape(preds,(len(preds),1))
    target = np.reshape(target,(len(preds),1))
    data = data.cpu().numpy()
    for i in range(len(preds)):
        if(preds[i]!=target[i]):
            pred_wrong.append(preds[i])
            true_wrong.append(target[i])
            incorrect_examples.append(data[i])

  return true_wrong, incorrect_examples, pred_wrong

def plot_incorrect_preds(true,ima,pred,n_figures = 10):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
    
    denorm = Normalize((-train_mean / train_std).tolist(), (1.0 / train_std).tolist())
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/5)
    fig,axes = plt.subplots(figsize=(12, 3), nrows = n_row, ncols=5)
    plt.subplots_adjust(hspace=1)
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)
        image,correct,wrong = ima[a],true[a],pred[a]
        image = torch.from_numpy(image)
        image = denorm(image)*255
        image = image.permute(2, 1, 0) # from NHWC to NCHW
        correct = int(correct)
        wrong = int(wrong)
        image = image.squeeze().numpy().astype(np.uint8)
        im = ax.imshow(image) #, interpolation='nearest')
        ax.set_title(f'A: {labels[correct]} , P: {labels[wrong]}', fontsize = 8)
        ax.axis('off')
    plt.show()
    
def plot_sample_imgs(train_loader,n_figures = 40):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
    ima, targets = next(iter(train_loader))
    denorm = Normalize((-train_mean / train_std).tolist(), (1.0 / train_std).tolist())
    n_row = int(n_figures/10)
    fig,axes = plt.subplots(figsize=(10, 3), nrows = n_row, ncols=10)
    plt.subplots_adjust(hspace=1)
    for ax in axes.flatten():
        a = random.randint(0,len(ima)-1)
        image, target = ima[a], targets[a]
#         image = torch.from_numpy(image)
        image = denorm(image)*255
        image = image.permute(2, 1, 0) # from NHWC to NCHW
        image = image.squeeze().numpy().astype(np.uint8)
        im = ax.imshow(image) #, interpolation='nearest')
        ax.set_title(f'{labels[target]}', fontsize = 8)
        ax.axis('off')
    plt.show()
    
def plot_gcam_incorrect_preds(model, true,ima,pred,n_figures = 10):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    
    denorm = Normalize((-train_mean / train_std).tolist(), (1.0 / train_std).tolist())
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/5)
    fig,axes = plt.subplots(figsize=(12, 3), nrows = n_row, ncols=5)
    plt.subplots_adjust(hspace=1)
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)
        image,correct,wrong = ima[a],true[a],pred[a]
        image = torch.from_numpy(image)
        grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=None) #torch.from_numpy(correct))
        grayscale_cam = grayscale_cam[0, :]

        image = denorm(image) #*255
        image = image.permute(2, 1, 0) # from NHWC to NCHW
        correct = int(correct)
        wrong = int(wrong)
        image = image.squeeze().numpy().astype(np.float32)
        image = np.clip(image, a_min = 0, a_max=1.)
#         im = ax.imshow(image) #, interpolation='nearest')
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        im = ax.imshow(visualization)
        ax.set_title(f'A: {labels[correct]} , P: {labels[wrong]}', fontsize = 8)
        ax.axis('off')
    plt.show()