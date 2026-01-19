# hawkes-repro

复现 “Spectral estimation of Hawkes processes from count data (2019)” 的仿真实验代码与结果。

## 摘要
本仓库包含复现论文中仿真实验的若干代码模块 和一个可交互 notebook。目标是（1）重现原论文的主要图表和数值；（2）提供可复现的代码供研究者参考；（3）探索使用神经网络改善whittle估计。

## 快速开始（最小示例）
1. 克隆仓库：
   git clone https://github.com/your-username/hawkes-repro.git
2. 进入目录并安装依赖（推荐使用 venv 或 conda）：
   pip install -r requirements.txt

## 目录结构
```
hawkes-repro/
│
│
├── results/               # 生成的图表和结果文件
│   ├── figure1.png        # 结果图表1
│   ├── figure2.png        # 结果图表2
│  
│
├── hawkes/                # 代码模块文件夹，包含算法实现
│   ├── MLE和Whittle简易演示.ipynb  # 简单演示最大似然与Whittle估计的Jupyter笔记本
│   ├── data_gener.py      # 数据生成模块，负责生成模拟数据
│   ├── mle_exp_plots.py   # MLE Exponential 估计图表
│   ├── mle_opt.py         # MLE 估计优化模块
│   ├── mle_power_plots.py # MLE Powerlaw 估计图表
│   ├── spectral.py        # Spectral估计相关算法实现
│   ├── whittle_exp_plots.py # Whittle Exponential 估计图表
│   ├── whittle_opt.py     # Whittle估计优化模块
│   └── whittle_power_plots.py # Whittle Powerlaw 估计图表
│
├── requirements.txt       # Python 依赖包列表
└── LICENSE                # 项目许可证
```

## 依赖与环境
见 `requirements.txt`。推荐 Python 3.9+。

## 许可与引用
本代码采用 MIT 许可证（见 LICENSE）。如在论文或工作中使用，请引用原论文：
> Author(s). Spectral estimation of Hawkes processes from count data. 2019.

如果你使用本仓库的实现或结果，也请引用本仓库：
> Zhang, M. (2025). hawkes-repro. https://github.com/your-username/hawkes-repro

## 联系
作者：章明 — mingjonnier[at]gmail.com  
GitHub: https://github.com/mingzhang4153.github.io
