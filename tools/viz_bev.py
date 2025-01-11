import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
# bev = torch.load('raybev.pt').detach().cpu().numpy().squeeze()
bev = torch.load('pointbev.pt').detach().cpu().numpy().squeeze()


bev_2d = np.where(np.all(bev == 0, axis=0), 0, 1)
plt.figure(figsize=(6, 6))
plt.imshow(bev_2d, cmap='gray')
plt.axis('off')
# plt.savefig('raybev.jpg')
plt.savefig('pointbev.jpg')


pdb.set_trace()