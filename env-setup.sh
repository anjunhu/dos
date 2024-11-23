conda create -n dos python=3.9.15
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# install from third party sources
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install third_party/Mask2Former/
pip install third_party/nvdiffrast/
pip install third_party/ODISE
# setup tiny-cuda-nn
cd third_party/tiny-cuda-nn/
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
python setup.py install
cd ../../../../
# install simple packages
pip install imagecodecs open-clip-torch neptune jaxtyping ipdb faiss-cpu==1.7.4 faiss-gpu 
pip install diffusers==0.27.2 ninja==1.11.1.1
pip install ../MVDream-threestudio/extern/MVDream/