from torchvision import datasets, transforms
train_dataset = datasets.MNIST(
   "../mnist_data",
   download=True,
   train=True,
   transform=transforms.Compose([transforms.ToTensor()]),
)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)