# %% [markdown]
# # Convolutional Neural Network

# %% [markdown]
# ### Libraries

# %%
# import libraries

import os
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from xml.etree import ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision.models import inception_v3, resnet50

# %%
# major variables

photos_dir = './photos'
renders_dir = './renders'

# %%
def parse_xml(xml_file):
    '''
    Read the xml file and return the bounding box coordinates
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bounding_boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bounding_boxes.append([xmin, ymin, xmax, ymax])
    return bounding_boxes

# %%
def load_data(data_dir):
    '''
    Returns a list of images and labels for each image
    '''
    image_paths = []
    num_legos = []
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                n = int(subdir.split(os.sep)[-1])
                image_paths.append(os.path.join(subdir, file))
                num_legos.append(n)
    combined = list(zip(image_paths, num_legos))
    combined.sort()
    image_paths, num_legos = zip(*combined)
    image_paths = np.asarray(image_paths)
    num_legos = torch.Tensor(num_legos).to(torch.int64)
    return image_paths, num_legos

# %%
# load data

image_paths, num_legos = load_data(photos_dir)

# %%
# work with defined train test split

train_test_split = np.genfromtxt('./train_test_split.csv', delimiter=',', dtype=None, encoding=None)

train_test_ids = {
    'train': [],
    'test': []
}
for index, row in enumerate(train_test_split):
    if row[1] == '1':
      train_test_ids['test'].append(index - 1)
    elif row[1] == '0':
      train_test_ids['train'].append(index - 1)

len(train_test_ids['train']), len(train_test_ids['test'])

# %%
# validation set

indices = train_test_ids['test']
np.random.shuffle(indices, )

test_size = 0.4 * len(indices)
split = int(np.floor(test_size))
train_test_ids['valid'], train_test_ids['test'] = indices[split:], indices[:split]

len(train_test_ids['train']), len(train_test_ids['valid']), len(train_test_ids['test'])

# %%
# class distribution in training data

num_legos_train = num_legos[train_test_ids['train']]
plt.hist(num_legos_train, bins=range(1, max(num_legos_train)), align='left', rwidth=0.8)
plt.xlabel('Number of Legos')
plt.ylabel('Frequency')
plt.title('Number of Legos Distribution')
plt.show()


# %%
class LegosDataset(Dataset):
    '''
    Dataset class for the legos dataset
    '''
    def __init__(self, images_filenames, num_legos, transform=None):
        self.images_filenames = images_filenames
        self.transform = transform
        self.labels = num_legos

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        label = self.labels[idx]
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# %%
# train, valid and test datasets

batch_size = 32
num_workers = 0

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_dataset = LegosDataset(image_paths[train_test_ids['train']], num_legos[train_test_ids['train']], transform=transform)
valid_dataset = LegosDataset(image_paths[train_test_ids['valid']], num_legos[train_test_ids['valid']], transform=transform)
test_dataset = LegosDataset(image_paths[train_test_ids['test']], num_legos[train_test_ids['test']], transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# %% [markdown]
# ### Model definition

# %%
# get cpu or gpu device for training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
# TODO: change definition

# %%
# put model in device

model = resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)


print(model)

# %% [markdown]
# ### Model training

# %%
def epoch_iter(dataloader, model, loss_fn, optimizer=None, is_train=True):
    '''
    Function for one epoch iteration
    '''
    if is_train:
        assert optimizer is not None, "When training, please provide an optimizer"
    num_batches = len(dataloader)
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.set_grad_enabled(is_train):
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            X, y = X.float().to(device), y.float().to(device)
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds.extend(pred.view(-1).cpu().detach().numpy())
            labels.extend(y.view(-1).cpu().numpy())
    return total_loss / num_batches, mean_squared_error(labels, preds)

# %%
def train(model, model_name, num_epochs, train_dataloader, validation_dataloader, loss_fn, optimizer):
    '''
    Function for training the model
    '''
    train_history = {'loss': [], 'MSE': []}
    val_history = {'loss': [], 'MSE': []}
    best_val_loss = np.inf
    print("Start training...")

    for t in range(num_epochs):
        print(f"Epoch {t+1}/{num_epochs}")
        train_loss, train_acc = epoch_iter(train_dataloader, model, loss_fn, optimizer)
        print(f"Train loss: {train_loss:.3f}, Train MSE: {train_acc:.3f}")
        val_loss, val_acc = epoch_iter(validation_dataloader, model, loss_fn, is_train=False)
        print(f"Validation loss: {val_loss:.3f}, Validation MSE: {val_acc:.3f}")

        # save model when val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': t
            }
            torch.save(save_dict, model_name + '_best_model.pth')

        # save latest model
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': t
        }
        torch.save(save_dict, model_name + '_latest_model.pth')

        # save training history
        train_history['loss'].append(train_loss)
        train_history['MSE'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['MSE'].append(val_acc)

    print("Finished")
    return train_history, val_history

# %%
# loss function

loss_fn = nn.MSELoss()

# %%
# learning rate

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
# train model

num_epochs = 25

train_history, val_history = train(model, 'lego_counter', num_epochs, train_dataloader, valid_dataloader, loss_fn, optimizer)

# %% [markdown]
# ### Training evolution analysis

# %%
def plotTrainingHistory(train_history, val_history):
    '''
    Plot the training history of the model
    '''
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(train_history['loss'], label='train')
    plt.plot(val_history['loss'], label='val')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Classification MSE')
    plt.plot(train_history['MSE'], label='train')
    plt.plot(val_history['MSE'], label='val')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

# %%
# visualize training history

plotTrainingHistory(train_history, val_history)

# %% [markdown]
# ### Model testing

# %%
# load best model

model = ().to(device)
checkpoint = torch.load('lego_counter_best_model.pth')
model.load_state_dict(checkpoint['model'])

# %%
# evaluate model on test data

test_loss, test_acc = epoch_iter(test_dataloader, model, loss_fn, is_train=False)
print(f"Test loss: {test_loss:.3f}, Test MSE: {test_acc:.3f}")

# %%
def show_predictions(model, dataloader):
    '''
    Display images along with their true and predicted labels
    '''
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu().numpy())
    for i in range(len(all_images)):
        plt.imshow(all_images[i].transpose((1, 2, 0)))
        plt.title(f'True label: {all_labels[i]}, Predicted label: {all_preds[i]}')
        plt.show()

# %%
# view predictions

show_predictions(model, test_dataloader)


