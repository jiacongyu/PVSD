# 全景视觉显著性检测（Panoramic Visual Saliency Detection）
<img src="./assets/images/t2.png" width="700" alt="投影方式对比">

## 引言
全景图像（ODI）数据具有360°×180°的视场范围，凭借其沉浸式体验优势，已在医疗、教育、娱乐等领域获得广泛关注。本综述系统梳理了2017-2025年深度学习技术在全景视觉显著性检测领域的最新进展，涵盖CNN、Transformer、LSTM等架构及多投影域建模方法。
我们创建此开源仓库，旨在为综述中提及的所有工作提供分类整理与代码链接，方便研究者快速追踪该领域的技术演进。我们将持续更新此仓库，尽可能收录最新研究成果，以期为全景视觉研究提供参考，并促进该领域研究社区的交流与发展。
## 全景图像（ODI）
### 1. 显著性预测(SP)
- **From Haziness to Clarity: A Novel Iterative Memory-Retrospective Emergence Model for Omnidirectional Image Saliency Prediction**  [Paper](https://ieeexplore.ieee.org/abstract/document/11045255/)
- **Multi-scale graph feature extraction network for panoramic image saliency detection**   [Paper](https://link.springer.com/article/10.1007/s00371-023-02825-x)

### 2. 显著性目标检测(SOD)
- **Consistency perception network for 360◦  omnidirectional salient object detection**  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224020149)
- **Breaking the Dataset Shackles: Data-Efficient Learning with Mamba Network for 360° Salient Object Detection**   [Paper](https://iopscience.iop.org/article/10.1088/1742-6596/3072/1/012004/meta)

## 全景视频（ODV）
### 1. 显著性预测(SP)
- **Spherical Vision Transformers for Audio-Visual Saliency Prediction in 360→ Videos**  [Paper](https://ieeexplore.ieee.org/abstract/document/11144923) [Code](https://cyberiada.github.io/SalViT360/)
- **CASP: Consistency-aware Audio-induced Saliency Prediction Model for Omnidirectional Video**  [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wan_CASP_Consistency-aware_Audio-induced_Saliency_Prediction_Model_for_Omnidirectional_Video_CVPR_2025_paper.html)
### 2. 显著性目标检测(SOD)
- **Instance-Level Panoramic Audio-Visual Saliency Detection and Ranking**  [Paper](https://dl.acm.org/doi/abs/10.1145/3664647.3681070)
- **PAV-SOD: A New Task towards Panoramic Audio visual Saliency Detection**  [Paper](https://dl.acm.org/doi/abs/10.1145/3565267)

## 全景显著性检测数据集
### 全景图像（ODI）
- **ODI-SOD** [dataset](https://github.com/iCVTEAM/ODI-SOD)
### 全景视频（ODV）
- **PAV-SOD** [dataset](https://github.com/ZHANG-Jun-Pu/PAV-SOD)
