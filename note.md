# ComfyUI_AdvancedReduxControl 项目概述

## 项目简介
该项目是 ComfyUI 的扩展，实现了对"Redux"模型和"Flux"模型的综合控制，允许用户对输入图像与文本提示（prompt）进行更灵活的:
- 融合
- 下采样
- 权重（strength）调整 
- 掩膜（mask）操作

项目采用 MIT 协议发布，核心功能集中在自定义节点（nodes）和工作流文件（.json）中。

## 主要文件结构
- **LICENSE**: MIT 开源协议
- **README.md**: 项目背景、使用方法与示例说明
- **advanced_workflow.json / simple_workflow.json**: 两种 ComfyUI 工作流文件
  - 高级可定制版
  - 简化基础版
- **nodes.py**: 核心 Python 代码，定义自定义节点与函数逻辑
- **init_.py**: 节点映射文件

## 工作流文件详解

### simple_workflow.json
展示基础功能流程:
- 加载模型与 VAE
- 随机噪声生成
- 采样与解码
- 图像保存
- 使用 StyleModelApplySimple 节点控制 Redux 影响强度（high/medium/low）

### advanced_workflow.json
在基础功能之上增添高级特性:
- ReduxAdvanced 节点支持:
  - 掩膜操作
  - 长宽比保持
  - 多种下采样方式（bicubic/area/nearest）
  - 阈值合并（token merging）
- 可调参数:
  - downsampling_factor
  - downsampling_function
  - autocrop_margin 等

## 自定义节点实现

### StyleModelApplySimple
- 通过强度枚举控制下采样倍率
- CLIP Vision 模型编码处理
- 图像与文本 token 融合

### ReduxAdvanced
高级功能节点，提供:
- 自定义下采样参数
- 掩膜处理
- 阈值合并
- CLIP Vision 27×27 token 编码
- 区域强调/忽略

### 辅助函数
- **automerge**: 相似度阈值 token 合并
- **apply_stylemodel**: 核心逻辑实现，处理图像 token 与文本 prompt 的合并