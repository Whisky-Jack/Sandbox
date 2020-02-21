import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleShapesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_frame.iloc[idx, 0])
        image = io.imread(img_name)
        labels = self.labels_frame.iloc[idx, 1:]
        #print(type(labels))
        #print(labels)
        #input()
        labels = np.array(labels, dtype=np.float16) #
        print(type(labels))
        print(labels)
        #print(labels)
        #input()
        #labels = labels.astype('object').reshape(-1, 2)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

"""
simple_shapes_dataset = SimpleShapesDataset(csv_file='./train/testFile.csv', root_dir = "./train",)

fig = plt.figure()

def imshow(img):
    #npimg = img.numpy()
    plt.imshow(img)

for i in range(len(simple_shapes_dataset)):
    sample = simple_shapes_dataset[i]
    print(type(sample['image']))
    print(sample['image'].shape)

    print(i, sample['image'].shape, sample['labels'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    imshow(sample['image'])


    if i == 3:
        plt.show()
        break
"""
class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'labels': labels}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'labels': labels}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image, (2, 0, 1))
        #print(type(labels))
        #print(labels)
        #input()
        return {'image': torch.from_numpy(image).float(),
                'labels': torch.from_numpy(labels)}
"""
class Normalize(object):

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = transforms.Normalize(mean, SD)
        #print(type(labels))
        #print(labels)
        #input()
        return {'image': torch.from_numpy(image),
                'labels': labels} #torch.from_numpy(labels)} #
"""
###############################################################
#helper functions
###############################################################
"""



###############################################################
#NEURAL NET SECTION
###############################################################
"""
#hyperparameters
n_epochs=2
batch_size_train=4 #data is returned in batches from the dataloader
batch_size_test=1000
learning_rate = 0.01
momentum=0.5


#scale = Rescale(80)
#crop = RandomCrop(40)
#tense = ToTensor()
#define operations required to preprocess the data
mean = [0.5, 0.5, 0.5] #normally this would be computed
SD = [0.5, 0.5, 0.5] #should also be computed

composed = transforms.Compose([RandomCrop(40),
                                Rescale(80),
                                ToTensor()])

#load datasets
trainset = SimpleShapesDataset(csv_file = './train/testFile.csv',
                                root_dir = "./train",
                                transform = composed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True)

testset = SimpleShapesDataset(csv_file = './test/testFile.csv',
                                root_dir = "./test",
                                transform = composed)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=False)
shape_classes = ("circle", "square", "triangle")
colour_classes = ("red", "blue", "green")

#################################################################
#IMAGE SHOWER
#################################################################

def imshow(images):
    #img = img*SD + mean     # unnormalize
    im_size = images.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images, nrow=3)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
nextSet = next(dataiter)
images, nextLabels = nextSet['image'], nextSet['labels']
#imshow(torchvision.utils.make_grid(images))
print(nextLabels)


#################################################################

#################################################################
#DEFINE NEURAL NETWORK
#################################################################
#define the neural network with some learnable parameters
#define the neural network with some learnable parameters
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 17 * 17, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)# 16*17*17)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net = net.float()
#################################################################

#define loss and optimization scheme
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#criterion =

#################################################################
#arrays for monitoring training
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)]

#################################################################
#TRAINING
#################################################################

for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the features; data is a list of [features, labels]
        features, shape, colour = data['image'], data['labels'][0], data['labels'][1]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(features)
        print(outputs.shape)
        print(data['labels'].shape)
        loss = F.nll_loss(outputs, data['labels'])#[shape, colour])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            train_losses.append(loss.item())
            train_counter.append(i*batch_size_train + len(trainloader.dataset)*(epoch-1))
            running_loss = 0.0

print('Finished Training')

#################################################################
#save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


#evaluate the network on some test data
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size_train)))

#load model back in
net = Net()
net.load_state_dict(torch.load(PATH))

#predict specific examples
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(batch_size_train)))


#################################################################
#TESTING
#################################################################
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

plt.plot(train_counter, train_losses)
plt.xlabel("Training Example Index")
plt.ylabel("Loss")
plt.show()
#################################################################


fig = plt.figure()

for i in range(1, len(dataset)):
    sample = dataset[i]

    print(i, sample['image'].shape, sample['labels'].shape)

    ax = plt.subplot(5, 4, i)
    plt.tight_layout()
    plt.imshow(sample['image'])

    if i ==20:
        plt.show()
        break
