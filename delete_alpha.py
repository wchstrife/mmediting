#coding=utf-8  
  
import os
path = '/home2/wangchenhao/mmediting/result/adobe1k/pred'

for file in os.listdir(path):
    os.rename(os.path.join(path,file), os.path.join(path, file[:-10] + '.png'))