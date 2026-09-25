# FlashH3VR

**Flash H3 Video Restoration**：在冻结的 H3 视频 VAE 内使用小型 Dense 残差适配器修复头部画面。

[English](README.md) · [推理说明](docs/USAGE.md) · [模型说明](MODEL_CARD.md) · [权重与依赖](docs/ASSETS.md) · [许可](THIRD_PARTY_NOTICES.md)

当前研发基线是 **Dense Inter 1837**，已在 256／448／640／832 四档训练。路径为冻结 H3 Encoder → 一次 Dense 潜空间残差修复 → 冻结 H3 Decoder，不使用 DiT、反复修复或 RGB NAF 尾部网络。

新的 **flashh3vr** 包负责当前推理。旧 h3ce 与 NAF 脚本保留为历史源码；旧版整帧回贴、24 FPS 数据及 16GB-VRAM 标签不能用于认证本模型。

## 安装

验证环境使用 Python 3.12、PyTorch 2.10.0。先安装适合本机 CUDA 的 PyTorch，再运行：

~~~bash
python -m pip install -e ".[inference]"
python -m flashh3vr --help
~~~

推理需分别准备指定的 H3 INT8 ConvRot 文件和 Dense Inter safetensors。H3 加载后走本轮训练对应的反量化 FP16 路径，不沿用旧版 Comfy Kitchen INT8 执行。文件 SHA、来源及当前下载状态见[权重与依赖](docs/ASSETS.md)，命令见[推理说明](docs/USAGE.md)。

## 本轮模型

- Dense Inter 有 3,152,128 个参数，导出四个 FP32 张量，约 12.0 MiB。
- 从 866 步研究起点新增 971 次更新，有效训练 105.005 分钟，完整 Adam 步数为 1837。
- 训练池含 75 张照片与 146 个真实视频窗；适配对象为 Go Youn-jung（高允贞／高胤祯），不是通用人像泛化认证。
- 使用原生 256 像素分块和至少 64 像素重叠；输入、输出均为已准备好的头部裁剪。
- 当前入口不提供整帧检测、人物跟踪、音频或回贴。

相对该轮起点，训练分区边缘误差下降 **8.08%**，复用开发验证下降 **4.10%**；448 照片嘴部误差回退约 **2.2%**。已审面板肉眼差异较小，未见新增明显严重伪影。结果不等于新独立泛化、长视频稳定性、FPS 或 16GB 部署认证。[数据口径与限制](docs/BASELINE.md)。

## 发布范围

项目自有源码沿用 **AGPL-3.0-only**，第三方源码保留原许可。H3 及关联权重的条款单独列明，不因源码开源而改为无限制许可。当前托管状态见[发布资产](docs/ASSETS.md)。Git 中不包含训练媒体、人物示例图、优化器、缓存或 H3 基座权重。

~~~bash
python -m pip install -e ".[dev,video,models,inference]"
python -m pytest -q
~~~

CPU 测试使用合成输入，不替代真实模型核验。项目独立维护，不代表 MiniMax 或素材人物认可。
