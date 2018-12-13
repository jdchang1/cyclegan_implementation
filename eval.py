import torch
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import *
from generator import ResNetGenerator

def eval(data_root, weights_1, weights_2):
    G_A2B = ResNetGenerator(3,3)
    G_B2A = ResNetGenerator(3,3)
    G_A2B.load_state_dict(torch.load(weights_1))
    G_B2A.load_state_dict(torch.load(weights_2))
    G_A2B.eval()
    G_B2A.eval()
    tsfms = [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    dataset = ImageDataset(data_root, tsfms=tsfms, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    input_A = torch.zeros([1,3,256,256]).type(torch.FloatTensor)
    input_B = torch.zeros([1,3,256,256]).type(torch.FloatTensor)
    for i, batch in enumerate(dataloader):
        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])
        fake_A = 0.5*(G_B2A(real_B).data + 1.0)
        fake_B = 0.5*(G_A2B(real_A).data + 1.0)
        save_image(fake_A, 'output/summer2winter/testA/{}.png'.format(i+1))
        save_image(fake_B, 'output/summer2winter/testB/{}.png'.format(i+1))

eval(sys.argv[1], sys.argv[2], sys.argv[3])
