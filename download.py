from torchvision import datasets, transforms
datasets.MNIST(root='./data', train=True, download=True)
datasets.MNIST(root='./data', train=False, download=True)
datasets.SVHN('./data',  download=True)
datasets.SVHN('./data', split='test', download=True)
datasets.CelebA('./data', download=True)
print('Done')