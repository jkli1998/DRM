## Installation

Most of the requirements of this projects are exactly the same as [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, we recommend you to check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python == 3.7
- PyTorch >= 1.7 (Mine 1.9.1 (CUDA 11.1))
- cocoapi
- yacs
- matplotlib
- OpenCV


### Step-by-step installation

```bash
# NOTE: we assume the python version is 3.7.x, if you want use other versions, please change the 4th line of `DRM/maskrcnn_benchmark/utils/imports.py` to corresponding python version. 

conda create --name DRM python=3.7
source activate DRM

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac
# IMPORTANT: here you need to change the 11-th line of apex/amp/_amp_state.py 
# to `if TORCH_MAJOR == 0 or TORCH_MINOR > 8:`
python setup.py install --cuda_ext --cpp_ext

# install mmcv
pip install -U openmim
mim install mmcv==1.7.0

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/jkli1998/DRM.git
cd DRM

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR

