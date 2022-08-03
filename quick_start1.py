# A toy example to show how to train transform network
import os
import torch
from model.transform_net import PlaneFinder
import torch.optim as optim

# set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# transformation network T
align_model = PlaneFinder(is_train=True)
align_model.train()
align_model.cuda()

optimizer = optim.AdamW(align_model.parameters(),
                        lr=1e-5, betas=(0.9, 0.999), weight_decay=5e-4)

# load CT data
# size: (batch_size, num_channel, num_slices, height, width)
x = torch.rand((4, 1, 40, 256, 256)).cuda()

optimizer.zero_grad()

# x_t: transformer symmetric x
# view: x, y, z rotation and translation
# M: transformation matrix
# please refer to the comments of forward function for the definition of each return variable
x_t, _, _, view, M, _ = align_model(x)

align_model.loss_total.backward()

optimizer.step()
