#!/usr/bin/python3

import sys
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from cyclegan import CycleGan
from utils import DiscriminatorBuffer, ImageDataset

def train(data_root, in_c=3, out_c=3, bs=1, num_epoch=200, lambda_idt=0.5, lambda_A=10.0, lambda_B=10.0, lambda_D=0.5):
    """
    in_c and out_c are 3 for the 3 color channels. These are left as
    variables to account for the potential to use grayscale pictures
    """
    dev_ = torch.device('cuda:0')

    gan = CycleGan(in_c, out_c)
    
    # Buffers with a size of 50 so that discriminator doesn't forget history
    A_buffer = DiscriminatorBuffer()
    B_buffer = DiscriminatorBuffer()
    
    # Data Augmentation
    tsfms = [transforms.Resize(int(256*1.12), Image.BICUBIC),
             transforms.RandomCrop(256),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    
    # Tensors
    goal_real = torch.ones([bs, 1, 15, 15], dtype=torch.float64, device=dev_, requires_grad=False).type(torch.FloatTensor).cuda()
    goal_fake = torch.zeros([bs, 1, 15, 15], dtype=torch.float64, device=dev_, requires_grad=False).type(torch.FloatTensor).cuda()
    input_A = torch.zeros([bs, in_c, 256, 256], dtype=torch.float64, device=dev_).type(torch.FloatTensor)
    input_B = torch.zeros([bs, out_c, 256, 256], dtype=torch.float64, device=dev_).type(torch.FloatTensor)
    
    # Dataset and Dataloader
    dataset = ImageDataset(data_root, tsfms=tsfms)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=6)
    print("Training Starting")
    # The training loop
    for epoch in range(1, num_epoch+1):
        running_g_loss = 0.0
        running_dA_loss = 0.0
        running_dB_loss = 0.0
        for i, batch in tqdm(enumerate(dataloader), desc="Epoch {}/{}".format(epoch, num_epoch)):
            real_A = input_A.copy_(batch['A']).cuda()
            real_B = input_B.copy_(batch['B']).cuda()
            # Generator outputs and update
            gan.optim_g.zero_grad()

            # Identity Loss
            idt_B = gan.G_A2B(real_B)
            idt_A = gan.G_B2A(real_A)
            idt_loss_B = gan.idt_criterion(idt_B, real_B)*lambda_B*lambda_idt
            idt_loss_A = gan.idt_criterion(idt_A, real_A)*lambda_A*lambda_idt

            # Generator GAN loss
            fake_B = gan.G_A2B(real_A)
            pred_B = gan.D_B(fake_B)
            fake_A = gan.G_B2A(real_B)
            pred_A = gan.D_A(fake_A)
            gan_loss_A2B = gan.gan_criterion(pred_B, goal_real)
            gan_loss_B2A = gan.gan_criterion(pred_A, goal_real)

            # Cycle Loss
            decode_A = gan.G_B2A(fake_B)
            decode_B = gan.G_A2B(fake_A)
            cyc_loss_ABA = gan.cycle_criterion(decode_A, real_A)*lambda_A
            cyc_loss_BAB = gan.cycle_criterion(decode_B, real_B)*lambda_B

            # Total Generator Loss
            g_loss = idt_loss_A + idt_loss_B + gan_loss_A2B + gan_loss_B2A + cyc_loss_ABA + cyc_loss_BAB
            g_loss.backward()
            gan.optim_g.step()

            # Discriminator outputs and update
            gan.optim_d.zero_grad()

            # Real Prediction Loss
            pred_real_A = gan.D_A(real_A)
            pred_real_B = gan.D_B(real_B)
            real_loss_A = gan.gan_criterion(pred_real_A, goal_real)
            real_loss_B = gan.gan_criterion(pred_real_B, goal_real)

            # Fake Prediction Loss
            fake_A = A_buffer.push_and_pop(fake_A)
            fake_B = B_buffer.push_and_pop(fake_B)
            pred_fake_A = gan.D_A(fake_A.detach())
            pred_fake_B = gan.D_B(fake_B.detach())
            fake_loss_A = gan.gan_criterion(pred_fake_A, goal_fake)
            fake_loss_B = gan.gan_criterion(pred_fake_B, goal_fake)

            loss_D_A = (real_loss_A + fake_loss_A)*lambda_D
            loss_D_B = (real_loss_B + fake_loss_B)*lambda_D
            loss_D_A.backward()
            loss_D_B.backward()
            gan.optim_d.step()

            running_g_loss += g_loss
            running_dA_loss += loss_D_A
            running_dB_loss += loss_D_B
        # Update Learning Rate according to scheduler
        gan.lr_update()
        print(running_g_loss, running_dA_loss, running_dB_loss)
        torch.save(gan.G_A2B.state_dict(), 'output/G_A2B.pt')
        torch.save(gan.G_B2A.state_dict(), 'output/G_B2A.pt')
        torch.save(gan.D_A.state_dict(), 'output/D_A.pt')
        torch.save(gan.D_B.state_dict(), 'output/D_B.pt')

train(sys.argv[1])    

