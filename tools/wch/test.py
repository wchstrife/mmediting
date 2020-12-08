import torch

# target = torch.randn([1,3,5,5])
# kx = torch.Tensor([[1, 0, -1], [2, 0, -2],[1, 0, -1]]).view(1, 1, 3, 3)
# ky = torch.Tensor([[1, 0, -1], [2, 0, -2],[1, 0, -1]]).view(1, 1, 3, 3).to(target)
# print(kx-ky)

# model = torch.load('/home/sensetime/work/mmediting/work_dirs/fba/model/R-50-GN-WS.pth')

# for key in model.keys():
#     print(key, model[key].size(), sep=" ")

# state_dict = {'encoder.' + k[7:]: v for k, v in model.items()}

# torch.save(state_dict, '/home/sensetime/work/mmediting/work_dirs/fba/model/resnet_50_GN_WS_rename.pth')

model = torch.load('/home/sensetime/work/mmediting/work_dirs/fba/model/iter_840000.pth')

temp = model.keys()

meta = model['meta']
state_dict = model['state_dict']
optimizer = model['optimizer']

for key in model.keys():
    print(key, model[key].size(), sep=" ")

