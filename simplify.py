import torch
from torch.nn import Sequential, Module
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import time
from PIL import Image, ImageStat
import argparse
import os

parser = argparse.ArgumentParser(description='Sketch simplification demo.')
parser.add_argument('--model', type=str, default='model_gan', help='Model to use.')
parser.add_argument('--inp', type=str, default=None, help='Input image filepath.')
parser.add_argument('--outp', type=str, default=None, help='Filepath to output.')
opt = parser.parse_args()

model_import = __import__(opt.model, fromlist=['model', 'immean', 'imstd'])
model = model_import.model
immean = model_import.immean
imstd = model_import.imstd

use_cuda = torch.cuda.device_count() > 0

# print(use_cuda)
model.load_state_dict(torch.load(opt.model + ".pth"))
model.eval()

torch.backends.cudnn.benchmark = True

imgs = os.listdir(opt.inp)
for img in imgs:
    start_time = time.time()
    print(f"Handling with {img}")
    img_path = os.path.join(opt.inp, img)

    data = Image.open(img_path).convert('L')
    size_origin = data.size
    data = data.resize((512,512))
    
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    stat = ImageStat.Stat(data)

    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
    #start_time =  time.time()
    if use_cuda:
        pred = model.cuda().forward(data.cuda()).float()
    else:
        pred = model.forward(data)
    #print('elapse time is {}'.format(time.time() - start_time))
    pred = F.interpolate(pred, size_origin)

    out_path = os.path.join(opt.outp, img)
    save_image(pred[0], out_path)
    print(f"time consumed: {time.time() - start_time}")
