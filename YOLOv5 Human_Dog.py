#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone repo')
get_ipython().system('pip install -qr yolov5/requirements.txt  # install dependencies (ignore errors)')
get_ipython().run_line_magic('cd', 'yolov5')

import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# In[20]:


# Export code snippet and paste here
get_ipython().run_line_magic('cd', '/content')
get_ipython().system('curl -L "https://public.roboflow.com/ds/CdfcQiUWX2?key=rLQRebgl5M" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip')


# In[22]:


# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
get_ipython().run_line_magic('cat', 'data.yaml')


# In[23]:


# define number of classes based on YAML
import yaml
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])


# In[24]:


#this is the model configuration we will use for our tutorial 
get_ipython().run_line_magic('cat', '/content/yolov5/models/yolov5s.yaml')


# In[25]:


#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


# In[26]:


get_ipython().run_cell_magic('writetemplate', '/content/yolov5/models/custom_yolov5s.yaml', "\n# parameters\nnc: {num_classes}  # number of classes\ndepth_multiple: 0.33  # model depth multiple\nwidth_multiple: 0.50  # layer channel multiple\n\n# anchors\nanchors:\n  - [10,13, 16,30, 33,23]  # P3/8\n  - [30,61, 62,45, 59,119]  # P4/16\n  - [116,90, 156,198, 373,326]  # P5/32\n\n# YOLOv5 backbone\nbackbone:\n  # [from, number, module, args]\n  [[-1, 1, Focus, [64, 3]],  # 0-P1/2\n   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4\n   [-1, 3, BottleneckCSP, [128]],\n   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8\n   [-1, 9, BottleneckCSP, [256]],\n   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16\n   [-1, 9, BottleneckCSP, [512]],\n   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32\n   [-1, 1, SPP, [1024, [5, 9, 13]]],\n   [-1, 3, BottleneckCSP, [1024, False]],  # 9\n  ]\n\n# YOLOv5 head\nhead:\n  [[-1, 1, Conv, [512, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 6], 1, Concat, [1]],  # cat backbone P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 13\n\n   [-1, 1, Conv, [256, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 4], 1, Concat, [1]],  # cat backbone P3\n   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)\n\n   [-1, 1, Conv, [256, 3, 2]],\n   [[-1, 14], 1, Concat, [1]],  # cat head P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)\n\n   [-1, 1, Conv, [512, 3, 2]],\n   [[-1, 10], 1, Concat, [1]],  # cat head P5\n   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)\n\n   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)\n  ]")


# In[27]:


# train yolov5s on custom data for 100 epochs
# time its performance
%%time
get_ipython().run_line_magic('cd', '/content/yolov5/')
get_ipython().system("python train.py --img 416 --batch 16 --epochs 100 --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache")


# In[28]:


# trained weights are saved by default in our weights folder
get_ipython().run_line_magic('ls', 'runs/')


# In[29]:


get_ipython().run_line_magic('ls', 'runs/train/exp0_yolov5s_results/weights')


# In[30]:


# when we ran this, we saw .007 second inference time. That is 140 FPS on a TESLA P100!
# use the best weights!
get_ipython().run_line_magic('cd', '/content/yolov5/')
get_ipython().system('python detect.py --weights runs/train/exp0_yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source ../test/images')


# In[31]:


#display inference on ALL test images
#this looks much better with longer training above

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/inference/output/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")


# In[13]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[14]:


get_ipython().run_line_magic('cp', '/content/yolov5/runs/exp0_yolov5s_results/weights/best.pt /content/gdrive/My\\ Drive')


# In[ ]:




