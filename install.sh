#!/bin/bash
#这个 install.sh 更像“作者当时环境依赖的安装清单”，不能严格称为完整、可靠的环境复现方案。
#它更接近：“在一个已经具有正确 Python + CUDA + PyTorch 基础环境的机器上，再把 DeepfakeBench 所需 Python 包补齐。”

pip install numpy==1.21.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==1.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Pillow==9.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install dlib==19.24.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install imageio==2.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install imgaug==0.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm==4.61.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy==1.7.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seaborn==0.11.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyyaml==6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install imutils==0.5.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python==4.6.0.66 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-image==0.19.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn==1.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install albumentations==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install efficientnet-pytorch==0.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==0.6.12 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install segmentation-models-pytorch==0.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchtoolbox==0.1.8.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard==2.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install setuptools==59.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install loralib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install filterpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install simplejson -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install kornia -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fvcore -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install imgaug==0.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install git+https://github.com/openai/CLIP.git
