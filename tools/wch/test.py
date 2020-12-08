import torch
import cv2

# target = torch.randn([1,3,5,5])
# kx = torch.Tensor([[1, 0, -1], [2, 0, -2],[1, 0, -1]]).view(1, 1, 3, 3)
# ky = torch.Tensor([[1, 0, -1], [2, 0, -2],[1, 0, -1]]).view(1, 1, 3, 3).to(target)
# print(kx-ky)

# model = torch.load('/home/sensetime/work/mmediting/work_dirs/fba/model/R-50-GN-WS.pth')

# for key in model.keys():
#     print(key, model[key].size(), sep=" ")

# state_dict = {'encoder.' + k[7:]: v for k, v in model.items()}

# torch.save(state_dict, '/home/sensetime/work/mmediting/work_dirs/fba/model/resnet_50_GN_WS_rename.pth')

# model = torch.load('work_dirs/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth')

# temp = model.keys()

# meta = model['meta']
# state_dict = model['state_dict']
# #optimizer = model['optimizer']

# for key in state_dict:
#     print(key, state_dict[key].size(), sep=" ")



img = cv2.imread('data/Adobe/merged/moon-jellyfish-aurelia-aurita-schirmqualle-66321.png')
img = cv2.resize(img, (320,320))
cv2.imwrite('data/Adobe/merged/fg.png', img)

img = cv2.imread('data/Adobe/trimap/moon-jellyfish-aurelia-aurita-schirmqualle-66321.png')
img = cv2.resize(img, (320,320), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('data/Adobe/trimap/trimap.png', img)