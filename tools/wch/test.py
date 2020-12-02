import torch

target = torch.randn([1,3,5,5])
kx = torch.Tensor([[1, 0, -1], [2, 0, -2],[1, 0, -1]]).view(1, 1, 3, 3)
ky = torch.Tensor([[1, 0, -1], [2, 0, -2],[1, 0, -1]]).view(1, 1, 3, 3).to(target)
print(kx-ky)