import imageio
import os
import argparse
import re
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='result/exp_4/png')
parser.add_argument('--output', type=str, default='result/exp_4/process.gif')

opt = parser.parse_args()

def add_title(image, title):
    width, height = image.size
    bi = Image.new('RGBA',(width+10, height+(height//5)),'white')
    bi.paste(image,(5,5,(width+5),(height+5)))
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    caption=title
    font = ImageFont.truetype(os.path.join('utils', "Microsoft-Sans-Serif.ttf"), 30)
    w, h = font.getsize(caption)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw=ImageDraw.Draw(bi)
    draw.text(((width-w)//2,(height+((height//5)-h)//2)),caption,font=font,fill='black')
    return bi

file_names = sorted(
    (os.path.join(opt.dir ,fn) for fn in os.listdir(opt.dir) if fn.endswith('.png')),
    key = lambda x: int(re.findall(r'\d+', x)[0])
)


images = []
for (i, fn) in enumerate(file_names):
    img = add_title(Image.open(fn), 'EPOCH %d' % (i+1))
    img = np.array(img)
    images.append(img)

filename = opt.output
imageio.mimsave(filename, images, duration = 0.5)