# 创建环境
conda create -n imfuse python=3.9
conda activate imfuse

# 安装PyTorch（根据CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install einops
pip install nibabel  # 用于处理nifti文件
pip install monai    # 医学图像处理工具
pip install wandb    # 实验跟踪
pip install tqdm     # 进度条

# 可选：安装更快的Mamba实现
pip install causal-conv1d>=1.1.0
pip install mamba-ssm