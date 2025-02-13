import numpy as np
import torch
import math
try:
    import comfy
    import folder_paths
    import nodes
    import os

    import re
    import safetensors
    import glob
except:
    pass
from collections import namedtuple

@torch.no_grad()
def automerge(tensor, threshold):
    """
    根据给定的相似性阈值 threshold 对张量 tensor 进行自动合并，以减少 token 总数。
    合并的依据是，如果当前 token 与上一个 token 的余弦相似度大于等于 threshold，则合并它们。
    
    Args:
        tensor (torch.FloatTensor): 形状为 (batchsize, slices, dim)。其中：
            batchsize 表示批大小，
            slices 表示序列长度（token 数量），
            dim 表示每个 token 的维度。
        threshold (float): 余弦相似度的合并阈值，取值范围为 [-1, 1]。
    
    Returns:
        torch.FloatTensor: 合并后的张量，形状仍然是 (batchsize, N, dim)，
        其中 N 视合并情况而定。
    """
    (batchsize, slices, dim) = tensor.shape
    newTensor=[]
    for batch in range(batchsize):
        tokens = []
        lastEmbed = tensor[batch,0,:]
        merge=[lastEmbed]
        tokens.append(lastEmbed)
        for i in range(1,slices):
            tok = tensor[batch,i,:]
            # 计算两个向量的余弦相似度
            cosine = torch.dot(tok, lastEmbed) / torch.sqrt(torch.dot(tok,tok) * torch.dot(lastEmbed,lastEmbed))
            if cosine >= threshold:
                # 如果大于给定阈值，则将该 token 与上一 token 合并（即取平均）
                merge.append(tok)
                lastEmbed = torch.stack(merge).mean(dim=0)
            else:
                # 否则将上一个 token 添加到tokens，并重置 merge
                tokens.append(lastEmbed)
                merge=[]
                lastEmbed = tok
        newTensor.append(torch.stack(tokens))
    return torch.stack(newTensor)

def automerge_v2(tensor, threshold):
    """
    改进版：在合并时保留位置权重，并采用累积权重的加权平均与归一化处理

    Args:
        tensor (torch.FloatTensor): 形状为 (batch, seq, dim)
        threshold (float): 合并 token 的余弦相似度阈值

    Returns:
        torch.FloatTensor: 合并后的 token 序列，形状为 (batch, N, dim)，其中每个 batch 内
                           的 N 可能不一致（按实际合并结果）。
    """
    batch, seq, dim = tensor.shape
    # 生成位置权重：线性从 1.0 衰减到 0.8（可根据需要调整）
    position_weights = torch.linspace(1.0, 0.8, steps=seq, device=tensor.device, dtype=tensor.dtype)
    
    merged_tokens = []
    for b in range(batch):
        tokens = []
        # 对于当前 batch，使用累积权重实现加权平均
        # 初始化第一个 token
        weighted_sum = tensor[b, 0] * position_weights[0]
        cumulative_weight = position_weights[0]
        merged_token = weighted_sum / cumulative_weight  # 初始合并 token

        for i in range(1, seq):
            current_weight = position_weights[i]
            current_weighted = tensor[b, i] * current_weight

            # 对当前合并 token 和当前 token 分别进行归一化，以便计算余弦相似度
            merged_norm = merged_token / (merged_token.norm(p=2) + 1e-8)
            current_norm = current_weighted / (current_weighted.norm(p=2) + 1e-8)
            cosine = torch.dot(merged_norm, current_norm)
            
            if cosine >= threshold:
                # 如果相似，更新累积权重和加权和
                weighted_sum = weighted_sum + current_weighted
                cumulative_weight += current_weight
                merged_token = weighted_sum / cumulative_weight
            else:
                # 如果当前 token与合并token相似度不够，则保存当前合并token，并重置累积信息
                tokens.append(merged_token)
                weighted_sum = current_weighted
                cumulative_weight = current_weight
                merged_token = weighted_sum / cumulative_weight
        # 别忘了将当前 batch 的最后一个合并 token 添加进来
        tokens.append(merged_token)
        merged_tokens.append(torch.stack(tokens))
    
    return torch.stack(merged_tokens)

# 这里定义了几个强度等级与对应的数值，主要用于简单的图像风格混合节点
STRENGTHS = ["highest", "high", "medium", "low", "lowest"]
STRENGTHS_VALUES = [1, 2, 3, 4, 5]

class StyleModelApplySimple:
    """
    一个简化版本的 StyleModel 应用节点，可在 ComfyUI 或类似环境中使用。
    通过一个枚举的"image_strength"参数来控制下采样强度，从而改变最终混合的效果。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "image_strength": (STRENGTHS, {"default": "medium"})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, image_strength):
        """
        将图像通过 CLIP Vision 模型编码得到的令牌（token），与 style_model 提供的风格向量混合进
        现有的 conditioning（文本提示）中，以实现图像风格辅助生成。
        
        Args:
            clip_vision_output (torch.FloatTensor): CLIP Vision 的输出张量（通常是图像特征）。
            style_model (Any): 包含 get_cond 方法的对象，用于进一步处理 vision 的输出。
            conditioning (list): 当前的文本条件列表，内部包含 (tokens, something) 形式的元素。
            image_strength (str): 决定下采样强度，可选值参照 STRENGTHS。
        
        Returns:
            tuple: 一个只包含一个元素的元组 (c,)，其中 c 为更新后的 conditioning 列表。
        """
        # 找到对应强度索引
        stren = STRENGTHS.index(image_strength)
        downsampling_factor = STRENGTHS_VALUES[stren]
        # 如果 factor==3 时使用 area，下采样平滑一些；否则使用 bicubic
        mode="area" if downsampling_factor==3 else "bicubic"
        
        # 将 style_model 的 cond 与 clip_vision_output 做进一步处理
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        # 下采样
        if downsampling_factor>1:
            (b,t,h) = cond.shape
            m = int(np.sqrt(t))
            cond = torch.nn.functional.interpolate(
                cond.view(b, m, m, h).transpose(1,-1),
                size=(m//downsampling_factor, m//downsampling_factor),
                mode=mode
            )
            cond = cond.transpose(1,-1).reshape(b, -1, h)
        
        # 将下采样后的 cond 拼接到每个 conditioning 元素中
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )

def standardizeMask(mask):
    """
    将输入的 mask 进行标准化，以满足后续处理需要。
    
    Args:
        mask (torch.FloatTensor or None): 输入的掩膜，可能是 2D 或 3D 张量。
    
    Returns:
        torch.FloatTensor or None: 标准化后的 4D 张量 (batch, 1, height, width)，或 None。
    """
    if mask is None:
        return None
    if len(mask.shape) == 2:
        (h,w)=mask.shape
        mask = mask.view(1,1,h,w)
    elif len(mask.shape)==3:
        (b,h,w)=mask.shape
        mask = mask.view(b,1,h,w)
    return mask

def crop(img, mask, box, desiredSize):
    """
    根据给定的 box 坐标与 desiredSize，先对图像进行插值缩放到 box 的大小，然后再裁剪。
    
    Args:
        img (torch.FloatTensor): 输入图像，形状 (batch, height, width, channels)。
        mask (torch.FloatTensor or None): 对应的掩膜。
        box (tuple): (ox, oy, w, h)，决定要如何截取图像。
        desiredSize (int): 目标尺寸，用于最终裁剪出 desiredSize × desiredSize 的图像。
    
    Returns:
        tuple: (img, mask)，分别是裁剪后的图像和掩膜。
    """
    (ox,oy,w,h) = box
    # 先将 mask 也做同样大小的缩放，如果存在 mask
    if mask is not None:
        mask = torch.nn.functional.interpolate(mask, size=(h,w), mode="bicubic").view(-1,h,w,1)
    # 将图像插值到给定的 h, w，然后裁剪到 desiredSize
    img = torch.nn.functional.interpolate(img.transpose(-1,1), size=(w,h), mode="bicubic", antialias=True)
    return (
        img[:, :, ox:(desiredSize+ox), oy:(desiredSize+oy)].transpose(1,-1),
        None if mask == None else mask[:, oy:(desiredSize+oy), ox:(desiredSize+ox),:]
    )

def letterbox(img, mask, w, h, desiredSize):
    """
    给图像和掩膜填充 letterbox（上下或左右留黑边），保证原始宽高比不变并调整到 desiredSize。
    在许多图像处理场景里，如果直接对图像进行裁剪或缩放，有时会破坏图像原本的长宽比。为保持图像纵横比不变，或填充成特定尺寸的方形输出，我们常用到一种被称为"letterbox"的处理方式。
    它会在图像的上下或者左右边缘添加留白（通常是黑边），这样既能保持原图的纵横比，又能使最终图像满足所需的输出分辨率（例如固定大小的方形）。
    在该函数（letterbox）中，其主要目的就是把图像缩放到指定宽高 w、h 后，再在较大的输出画布 desiredSize×desiredSize 上进行居中贴合。
    如果图像并不是正方形，就会在上下或左右自动生成黑色填充区域；在需要掩膜（mask）时，对掩膜也实施同样的填充操作，从而使图像与掩膜在空间尺寸上保持一致。
    
    Args:
        img (torch.FloatTensor): 输入图像，形状 (batch, height, width, channels)。
        mask (torch.FloatTensor or None): 掩膜。
        w (int): 缩放后的目标宽度。
        h (int): 缩放后的目标高度。
        desiredSize (int): 需要的输出方形大小。
    
    Returns:
        tuple: (img, mask)，分别是填充并调整到 desiredSize 的图像与掩膜。
    """
    (b,oh,ow,c) = img.shape
    # 首先插值到 w,h
    img = torch.nn.functional.interpolate(img.transpose(-1,1), size=(w,h), mode="bicubic", antialias=True).transpose(1,-1)
    
    # 然后在一个 (desiredSize, desiredSize) 的零张量上进行贴合（letterbox）
    letterbox = torch.zeros(size=(b, desiredSize, desiredSize, c))
    offsetx = (desiredSize - w) // 2
    offsety = (desiredSize - h) // 2
    letterbox[:, offsety:(offsety+h), offsetx:(offsetx+w), :] += img
    img = letterbox
    
    # 如果有掩膜，也要做相同操作
    if mask is not None:
        mask = torch.nn.functional.interpolate(mask, size=(h,w), mode="bicubic")
        letterbox = torch.zeros(size=(b,1,desiredSize,desiredSize))
        letterbox[:, :, offsety:(offsety+h), offsetx:(offsetx+w)] += mask
        mask = letterbox.view(b,1,desiredSize,desiredSize)
    return (img, mask)

def getBoundingBox(mask, w, h, relativeMargin, desiredSize):
    """
    在掩膜 mask 中搜索前景部分的最小边界矩形，并根据 relativeMargin 进行适当放大，
    同时保证最终截取宽高不会超出图像。若最终边界区域小于 desiredSize，则会进行相应扩展。
    
    Args:
        mask (torch.FloatTensor): 掩膜 (1, h, w) 或 (b, h, w)。
        w (int): 图像宽度。
        h (int): 图像高度。
        relativeMargin (float): 扩张边界的相对比例。
        desiredSize (int): 最小保留尺寸，用于保留感兴趣区域。
    
    Returns:
        (int, int, int, int): 截取区域的四个值 (x_min, y_min, x_max, y_max)。
    """
    mask = mask.view(h, w)
    marginW = math.ceil(relativeMargin * w)
    marginH = math.ceil(relativeMargin * h)
    indices = torch.nonzero(mask, as_tuple=False)
    y_min, x_min = indices.min(dim=0).values
    y_max, x_max = indices.max(dim=0).values
    
    x_min = max(0, x_min.item() - marginW)
    y_min = max(0, y_min.item() - marginH)
    x_max = min(w, x_max.item() + marginW)
    y_max = min(h, y_max.item() + marginH)
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    larger_edge = max(box_width, box_height, desiredSize)
    # 如果当前检测到的边界小于 larger_edge，则适当向上下左右扩展
    if box_width < larger_edge:
        delta = larger_edge - box_width
        left_space = x_min
        right_space = w - x_max
        expand_left = min(delta // 2, left_space)
        expand_right = min(delta - expand_left, right_space)
        # 再次比对剩余空间，如果有剩余再进行一次扩展
        expand_left += min(delta - (expand_left + expand_right), left_space - expand_left)
        x_min -= expand_left
        x_max += expand_right

    if box_height < larger_edge:
        delta = larger_edge - box_height
        top_space = y_min
        bottom_space = h - y_max
        expand_top = min(delta // 2, top_space)
        expand_bottom = min(delta - expand_top, bottom_space)
        expand_top += min(delta - (expand_top + expand_bottom), top_space - expand_top)
        y_min -= expand_top
        y_max += expand_bottom

    # 最终确保不越界
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    return x_min, y_min, x_max, y_max

def patchifyMask(mask, patchSize=14):
    """
    将输入的 mask 根据给定的 patchSize 进行分块（类似于 ViT 的 token 分区）。
    主要用于在进行 token 化时，对于哪些区域应该被"激活"或"忽略"进行控制。
    
    Args:
        mask (torch.FloatTensor or None): 形状通常为 (batch, size, size, 1)。
        patchSize (int): 每个 patch 的大小。
    
    Returns:
        torch.FloatTensor or None: 分块后的 mask；若输入为 None 则返回 None。
    """
    if mask is None:
        return mask
    (b, imgSize, imgSize, _) = mask.shape
    toks = imgSize // patchSize
    # 使用 MaxPool2d 来对 mask 进行简化，使得一个 patch 内如果有激活，则整体激活。
    return torch.nn.MaxPool2d(kernel_size=(patchSize,patchSize), stride=patchSize)(
        mask.view(b,imgSize,imgSize)
    ).view(b,toks,toks,1)

def prepareImageAndMask(visionEncoder, image, mask, mode, autocrop_margin, desiredSize=384):
    """
    根据 mode 选择不同的图像处理方式，将 image 和 mask 处理成期望大小（等比缩放、居中裁剪或自动裁剪）。
    
    Args:
        visionEncoder: CLIP 等视觉编码器，可在后续对图像进行编码（此处未使用，仅传入以保持接口一致）。
        image (torch.FloatTensor): 形状 (B, H, W, C) 的图像数据。
        mask (torch.FloatTensor or None): 掩膜。
        mode (str): 具体处理模式，参照 IMAGE_MODES。
        autocrop_margin (float): 在自动裁剪模式下，对边界区域的额外留白比例。
        desiredSize (int): 目标输出大小。
    
    Returns:
        (torch.FloatTensor, torch.FloatTensor or None): 处理后的图像和掩膜。
    """
    mode = IMAGE_MODES.index(mode)
    (B,H,W,C) = image.shape

    if mode == 0:  # center crop (square)
        imgsize = min(H, W)
        ratio = desiredSize / imgsize
        (w, h) = (round(W * ratio), round(H * ratio))
        # 中心裁剪时，会先全图缩放，再从中心提取 desiredSize
        image, mask = crop(image, standardizeMask(mask), ((w - desiredSize)//2, (h - desiredSize)//2, w, h), desiredSize)

    elif mode == 1:  # keep aspect ratio
        if mask is None:
            mask = torch.ones(size=(B,H,W))
        imgsize = max(H, W)
        ratio = desiredSize / imgsize
        (w, h) = (round(W * ratio), round(H * ratio))
        image, mask = letterbox(image, standardizeMask(mask), w, h, desiredSize)

    elif mode == 2:  # autocrop with mask
        (bx, by, bx2, by2) = getBoundingBox(mask, W, H, autocrop_margin, desiredSize)
        image = image[:, by:by2, bx:bx2, :]
        mask = mask[:, by:by2, bx:bx2]
        imgsize = max(bx2-bx, by2-by)
        ratio = desiredSize / imgsize
        (w, h) = (round((bx2 - bx)*ratio), round((by2 - by)*ratio))
        image, mask = letterbox(image, standardizeMask(mask), w, h, desiredSize)

    return (image, mask)

def processMask(mask, imgSize=384, patchSize=14):
    """
    先对输入 mask 进行缩放到约定的大小 imgSize，然后再按某个 patchSize 进行分块。
    返回的结果通常用于在进一步 token 化时决定要保留或者忽略哪些 patch。
    
    Args:
        mask (torch.FloatTensor): 可能是 2D / 3D。
        imgSize (int): 缩放后统一的图像大小。
        patchSize (int): 分块大小。
    
    Returns:
        torch.FloatTensor: 最终分块后的掩膜。
    """
    if len(mask.shape) == 2:
        (h,w)=mask.shape
        mask = mask.view(1,1,h,w)
        b=1
    elif len(mask.shape)==3:
        (b,h,w)=mask.shape
        mask = mask.view(b,1,h,w)
    scalingFactor = imgSize / min(h,w)
    # 首先插值到新的大小
    mask = torch.nn.functional.interpolate(
        mask, size=(round(h*scalingFactor), round(w*scalingFactor)), mode="bicubic"
    )
    # 进行居中裁剪，使其大小恰好是 (imgSize,imgSize)
    horizontalBorder = (imgSize - mask.shape[3]) // 2
    verticalBorder = (imgSize - mask.shape[2]) // 2
    mask = mask[:, :, verticalBorder:(verticalBorder+imgSize), horizontalBorder:(horizontalBorder+imgSize)].view(b, imgSize, imgSize)
    
    # 最后根据 patchSize 进行分块
    toks = imgSize // patchSize
    return torch.nn.MaxPool2d(kernel_size=(patchSize,patchSize), stride=patchSize)(mask).view(b,toks,toks,1)

IMAGE_MODES = [
    "center crop (square)",
    "keep aspect ratio",
    "autocrop with mask",
    "multi-slice"  # 新增模式
]

class ReduxAdvanced:
    """
    高级版本的 StyleModel 节点，提供了更多的自定义参数，
    包括下采样因子 downsampling_factor、下采样插值方式 downsampling_function、
    可选的掩膜 mask、自动裁剪、及进一步的权重控制等，以实现更灵活的图像-文本融合调优。
    """

    """
    Nearest (最近邻插值):本质: 选择距离目标位置最近的像素的值作为插值结果。
    特点: 计算简单，速度快，但可能会导致图像出现锯齿状（阶梯效应），特别是在放大图像时。
    适用场景: 适用于不需要平滑处理的场景，如二值图像或对速度要求较高的情况。

    Bilinear (双线性插值):
    本质: 在两个方向上进行线性插值，计算目标位置周围四个最近邻像素的加权平均值。
    特点: 计算复杂度适中，插值结果较为平滑，适用于大多数图像处理任务。
    适用场景: 广泛应用于图像缩放、旋转等场景，能够较好地保持图像细节。

    Bicubic (双三次插值):
    本质: 在两个方向上进行三次多项式插值，计算目标位置周围16个最近邻像素的加权平均值。
    特点: 插值结果非常平滑，能够较好地保持图像细节，但计算复杂度较高。
    适用场景: 适用于需要高质量图像缩放的场景，如图像增强、细节保留等。

    Area (区域插值):
    本质: 计算目标位置周围像素的平均值作为插值结果。
    特点: 计算简单，速度快，但可能会导致图像细节丢失。
    适用场景: 适用于不需要保留细节的场景，如图像压缩、模糊处理等。

    Nearest-exact (精确最近邻插值):
    本质: 选择距离目标位置最近的像素的值作为插值结果，但要求目标位置必须是整数坐标。
    特点: 计算简单，速度快，但可能会导致图像出现锯齿状（阶梯效应），特别是在放大图像时。
    适用场景: 适用于不需要平滑处理的场景，如二值图像或对速度要求较高的情况。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision": ("CLIP_VISION", ),
                             "image": ("IMAGE",),
                             "downsampling_factor": ("INT", {"default": 3, "min": 1, "max":9}),
                             "downsampling_function": (["nearest", "bilinear", "bicubic","area","nearest-exact"], {"default": "area"}),
                             "mode": (IMAGE_MODES, {"default": "center crop (square)"}),
                             "weight": ("FLOAT", {"default": 1.0, "min":0.0, "max":1.0, "step":0.01}),
                             "slice_version": (["v1", "v2"], {"default": "v2"}),
                             "interpolation_mode": (["nearest", "bilinear", "bicubic"], {"default": "bicubic"}),
                             "merge_threshold": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01})
                            },
                "optional": {
                            "mask": ("MASK", ),
                            "autocrop_margin": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01})
                }}
    RETURN_TYPES = ("CONDITIONING","IMAGE", "MASK")
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision, image, style_model, conditioning,
                         downsampling_factor, downsampling_function, mode, weight,
                         mask=None, autocrop_margin=0.0, slice_version="v2", interpolation_mode="bicubic", merge_threshold=0.95):
        """
        将输入图像先用 CLIP Vision 编码出特征，再由 style_model 将该特征转换为可融合至文本条件中的向量。
        可以通过多种下采样方式、掩膜等手段精细控制这些向量的生成与合并。
        
        Args:
            clip_vision (Any): 通常是 CLIP 模型或者兼容的视觉编码器。
            image (torch.FloatTensor): 原始图像 (B, H, W, C)。
            style_model (Any): 具备 get_cond 方法的对象，用于处理视觉特征。
            conditioning (list): 当前的文本提示，内部包含若干 (token, something)。
            downsampling_factor (int): 将特征图下采样多少倍。
            downsampling_function (str): 选择插值方式（如 "bicubic"、"area" 等）。
            mode (str): "center crop (square)" / "keep aspect ratio" / "autocrop with mask" / "multi-slice"。
            weight (float): 融合权重，越大则该图像特征在最终生成中的影响力越大。
            mask (torch.FloatTensor or None): 掩膜，用于指定关注的图像区域。
            autocrop_margin (float): 在 "autocrop with mask" 时，给前景区域额外留出的边距。
            slice_version (str): 选择切片版本，"v1" 或 "v2"。
            interpolation_mode (str): 插值算法，"nearest"、"bilinear" 或 "bicubic"。
            merge_threshold (float): 视觉 token 合并的相似性阈值，取值范围为 [0.5, 1.0]。
        
        Returns:
            (list, torch.FloatTensor, torch.FloatTensor or None): 
            分别是更新后的 conditioning、处理过的图像、以及部分处理后的掩膜(若 masko 不为 None)。
        """
        # 根据 mode 对图像和掩膜进行一系列预定义的裁剪和缩放
        """
         根据指定的 mode（图像处理模式），对输入的 image 和 mask 进行裁剪和缩放，确保它们符合后续处理的要求。

         1. 如果 mode 为 "center crop (square)"，则先计算图像的宽高比，然后根据 desiredSize 裁剪出正方形图像。
         2. 如果 mode 为 "keep aspect ratio"，则先计算图像的宽高比，然后根据 desiredSize 缩放图像，保持宽高比不变。
         3. 如果 mode 为 "autocrop with mask"，则先计算图像的宽高比，然后根据 desiredSize 裁剪出正方形图像。
        """
        # 修改后的图像预处理部分
        if mode != "multi-slice":  # 新模式下跳过常规预处理
            image, masko = prepareImageAndMask(clip_vision, image, mask, mode, autocrop_margin)
        else:  # 新模式直接使用原始图像
            masko = None
        """
        image 经过处理变成 torch.Size([1, 384, 384, 3])
        """
        
        # 用 CLIP 对图像进行编码，得到图像特征；将掩膜进一步 patchify
        """
        使用 CLIP Vision 模型将预处理后的图像编码为特征向量。
        掩膜的分块处理（patchify）是为了与特征向量的 token 对应起来，便于在下游任务中应用掩膜
        """
        clip_vision_output, mask = (clip_vision.encode_image(image), patchifyMask(masko))
        """
        clip_vision.encode_image(image) 通过siglip模型将像素图像编码成视觉token，张量为: torch.Size([1, 729, 1152])
        """
        
        # 这里将 style_model.get_cond(clip_vision_output) 得到的视觉向量视为 cond
        # shape 通常为 (batch, token, hidden_dim)
        mask_mode="area"  # 此处可视为固定最终 mask 插值方式；实际插值方式受 downsampling_function 控制

        """
        style_model.get_cond是将clip_vision_output输入到style_model中，得到cond
        其中style_model 的模型定义以及前向传播代码如下
        class ReduxImageEncoder(torch.nn.Module):
            def __init__(
                self,
                redux_dim: int = 1152,
                txt_in_features: int = 4096,
                device=None,
                dtype=None,
            ) -> None:
                super().__init__()

                self.redux_dim = redux_dim
                self.device = device
                self.dtype = dtype

                self.redux_up = ops.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
                self.redux_down = ops.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

            def forward(self, sigclip_embeds) -> torch.Tensor:
                projected_x = self.redux_down(torch.nn.functional.silu(self.redux_up(sigclip_embeds)))
                return projected_x

        以此可以得知style_model.get_cond(clip_vision_output) 的输出为 (batch, token, hidden_dim)
        接着flatten操作会使得cond的shape变为(batch*token, hidden_dim)
        最后unsqueeze(dim=0) 会使得cond的shape变为(1, batch*token, hidden_dim)
        这么做的目的是为了将cond视为一个整体，便于后续的mask操作
        """
        # 修改后的 multi-slice 处理逻辑
        if mode == "multi-slice":
            if slice_version == "v1":
                slices = generate_slices(image, base_size=384, overlap_ratio=0.1)
            else:  # v2版本
                slices = generate_slices_v2(image, 
                    base_size=384,
                    interpolation_mode=interpolation_mode)
            
            cond_slices = []
            
            # 处理每个切片
            for slice_img, _ in slices:
                # 1. 视觉编码
                clip_vision_output = clip_vision.encode_image(slice_img)
                # 3. 通过 style_model 转换条件
                cond_slice = style_model.get_cond(clip_vision_output)  # 形状保持 [1, num_tokens, txt_dim]
                cond_slices.append(cond_slice)
            
            # 4. 沿 token 维度拼接所有切片条件
            cond = torch.cat(cond_slices, dim=1)  # 最终形状 [1, total_tokens, txt_dim]
            mask = None
            
            # 保留切片图像用于返回
            slice_tensors = [i[0] for i in slices]
            image = torch.cat(slice_tensors, dim=0)
        else:
            # 常规模式处理
            clip_vision_output = clip_vision.encode_image(image)
            mask = patchifyMask(masko)
            cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)

        (b, t, h) = cond.shape

        # 这里的m是为了将cond的shape变为(1, m, m, hidden_dim)
        m = int(np.sqrt(t))
        
        # 如果需要下采样，就把原本 (1, batch*token, hidden_dim) 视为 (1, m, m, hidden_dim) 再插值
        if downsampling_factor>1:
            cond = cond.view(b, m, m, h)
            if mask is not None:
                # 将cond与mask相乘，使得mask中为0的区域不参与插值
                cond = cond * mask
            # 使用指定的 downsampling_function（如 "area"）对 cond 进行下采样，减少 token 数量
            cond = torch.nn.functional.interpolate(
                cond.transpose(1,-1),
                size=(m//downsampling_factor, m//downsampling_factor),
                mode=downsampling_function
            )
            # 将 cond 的形状从 (1, m//downsampling_factor, m//downsampling_factor, h) 转换为 (b, -1, h)
            cond = cond.transpose(1,-1).reshape(b, -1, h)
            # 同时对 mask 也完成对应大小的插值
            mask = None if mask is None else torch.nn.functional.interpolate(
                mask.view(b, m, m, 1).transpose(1,-1),
                size=(m//downsampling_factor, m//downsampling_factor),
                mode=mask_mode
            ).transpose(-1,1)
        
        #  将条件向量 cond 乘以权重平方，以调整其在最终条件中的影响力度
        # 不直接使用线性权重（cond * weight），而是使用平方权重（cond * weight * weight），
        # 这样可以确保权重在最终条件中的影响力度是线性的，而不是平方的。
        cond = cond * (weight * weight)
        
        # 利用 merge_threshold 对相似的视觉 token 进行自动合并
        # 当 merge_threshold 小于 1.0 时执行合并，否则保留所有 token
        if merge_threshold < 1.0:
            print(f"[DEBUG] Pre-merge cond shape: {cond.shape}") 
            cond = automerge_v2(cond, merge_threshold)
            print(f"[DEBUG] Post-merge cond shape: {cond.shape}")
        
        # 如果存在 mask，我们会先用 mask 筛选 cond，再填充到相同长度
        if mask is not None:
            # 将 mask 转换为二进制掩膜，并 reshape 为 (b, -1) 的形状
            mask = (mask>0).reshape(b, -1)
            # 计算每个样本中 mask 中非零元素的最大数量
            max_len = mask.sum(dim=1).max().item()
            # 创建一个形状为 (b, max_len, h) 的零张量，用于存储填充后的条件向量
            padded_embeddings = torch.zeros((b, max_len, h), dtype=cond.dtype, device=cond.device)
            for i in range(b):
                # 筛选出 mask 中非零的 cond 向量
                filtered = cond[i][mask[i]]
                # 将筛选后的 cond 向量填充到 padded_embeddings 中   
                padded_embeddings[i, :filtered.size(0)] = filtered
            # 将填充后的条件向量赋值给 cond
            cond = padded_embeddings
        
        # 最终，将 cond 拼接回原先的 conditioning，以产生综合 prompt
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, image, masko)

def generate_slices(image_tensor, base_size=384, overlap_ratio=0.1):
    """
    对输入图像张量进行自适应切片，生成多个 base_size x base_size 的切片
    
    Args:
        image_tensor: (B, H, W, C) 格式的输入图像
        base_size: 切片基准尺寸
        overlap_ratio: 切片重叠率
    
    Returns:
        slices: 切片列表 [(slice_tensor, (x1, y1, x2, y2)), ...]
    """
    B, H, W, C = image_tensor.shape
    stride = int(base_size * (1 - overlap_ratio))
    slices = []
    
    # 自动计算切片数量
    num_rows = (H - base_size) // stride + 1
    num_cols = (W - base_size) // stride + 1
    
    for y in range(0, num_rows * stride + 1, stride):
        for x in range(0, num_cols * stride + 1, stride):
            y_end = min(y + base_size, H)
            x_end = min(x + base_size, W)
            
            # 边缘处理：当剩余空间不足时反向偏移
            if y_end - y < base_size:
                y = max(0, y_end - base_size)
            if x_end - x < base_size:
                x = max(0, x_end - base_size)
                
            slice_tensor = image_tensor[:, y:y_end, x:x_end, :]
            slices.append((slice_tensor, (x, y, x_end, y_end)))
    
    return slices


def generate_slices_v2(image_tensor, base_size=384, interpolation_mode="bicubic"):
    """
    改进版切片函数，先对齐分辨率再切片
    
    Args:
        image_tensor: (B, H, W, C) 格式的输入图像
        base_size: 切片基准尺寸
        interpolation_mode: 插值算法
    
    Returns:
        slices: 切片列表 [(slice_tensor, (x1, y1, x2, y2)), ...]
    """
    B, H, W, C = image_tensor.shape
    
    # 分辨率对齐
    def align_resolution(size):
        return max(base_size, (size // base_size) * base_size)
    
    aligned_H = align_resolution(H)
    aligned_W = align_resolution(W)
    
    # 使用指定插值算法缩放
    if aligned_H != H or aligned_W != W:
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.permute(0, 3, 1, 2),
            size=(aligned_H, aligned_W),
            mode=interpolation_mode,
            antialias=True
        ).permute(0, 2, 3, 1)
    
    # 生成规则切片
    slices = []
    num_rows = aligned_H // base_size
    num_cols = aligned_W // base_size
    
    for row in range(num_rows):
        for col in range(num_cols):
            y = row * base_size
            x = col * base_size
            slice_tensor = image_tensor[:, y:y+base_size, x:x+base_size, :]
            slices.append((slice_tensor, (x, y, x+base_size, y+base_size)))
    
    return slices

class ReduxAdvancedPaligemma:
    """
    此节点采用 paligemma2-3b-pt-896 模型来编码图像，从而在保留细节的同时维持整体结构。
    
    输入参数说明：
        conditioning: 文本编码器生成的 conditioning，格式为列表，每个元素为 [tensor, options_dict]
        style_model: 风格模型，需包含 get_cond 方法，将 clip_vision 的输出转换为视觉特征向量。
        clip_vision: CLIP 或兼容的视觉编码器，负责将图像转换为视觉特征。
        image: 输入图像，格式为 torch.Tensor，其形状为 (B, H, W, C)。
        resize: 整数，支持的取值为 224, 384, 448, 896，用于在将图像传入 clip_vision 前对图像进行缩放，
                缩放目标为正方形尺寸 (resize, resize)。
                
    返回：
        (CONDITIONING, IMAGE)：
            CONDITIONING - 将文本条件与视觉特征融合后的结果，格式和各下游模块兼容；
            IMAGE - 缩放后的图像，用于后续处理或展示。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "style_model": ("STYLE_MODEL", ),
                "clip_vision": ("CLIP_VISION", ),
                "image": ("IMAGE", ),
                "resize": ("INT", {"default": 384, "options": [224, 384, 448, 896]})
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "IMAGE")
    FUNCTION = "apply_stylemodel"
    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision, image, style_model, conditioning, resize):
        """
        根据指定的 resize 参数调整输入 image 的尺寸，再利用新的 paligemma2-3b-pt-896 模型进行视觉编码，
        最后将得到的视觉 tokens 与文本条件进行拼接，返回融合后的 conditioning 以及处理后的 image。
        """
        import torch.nn.functional as F

        # 假设输入 image 形状为 (B, H, W, C)，先转换为 (B, C, H, W) 方便插值
        image_t = image.permute(0, 3, 1, 2)
        # 按照 resize 指定的尺寸重缩放得到目标大小的图像
        image_resized = F.interpolate(image_t, size=(resize, resize), mode="bicubic", align_corners=False)
        # 转换回 (B, H, W, C)
        image_resized = image_resized.permute(0, 2, 3, 1)
        
        # 使用 clip_vision 对缩放后的图像进行编码
        clip_vision_output = clip_vision.encode_image(image_resized)
        # 利用 style_model 提取视觉特征（视觉 tokens），格式一般为 (B, tokens, hidden_dim)
        cond = style_model.get_cond(clip_vision_output)
        
        # 将视觉 token flatten 成合适的形状，形式转换为 (1, batch*tokens, hidden_dim)
        cond = cond.flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        # 将视觉 tokens 拼接回原先的文本 conditioning 中，构成综合的提示信息
        c = []
        for t in conditioning:
            # 假设 t[0] 是文本 token，t[1] 是附加信息（例如 pooled_output 或 guidance）
            new_cond = torch.cat((t[0], cond), dim=1)
            c.append([new_cond, t[1].copy()])
        
        # 返回融合后的 conditioning 和缩放后的图像
        return (c, image_resized)


class DuplicateConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "copy_factor": ("INT", {"default": 0, "min": 0, "max": 20}),
                "mode": (["空白模式", "本体模式"], {"default": "本体模式"})
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "duplicate_conditioning"
    CATEGORY = "conditioning/utils"

    def duplicate_conditioning(self, conditioning, copy_factor, mode):
        """
        节点功能：
          - 根据输入的模式生成新的 conditioning token：
              * "空白模式"：创建与原 conditioning 同形状但数值全部为 0 的张量；
              * "本体模式"：直接复制原 conditioning 张量。
          - 如果 copy_factor 为 0 则不做任何处理，直接返回原始 conditioning；
          - 否则，将得到的新 token 根据复制因子 copy_factor 复制多份，
            然后沿 token 的维度（dim=1）拼接在原 conditioning 后面，
            并存储到新的 conditioning 列表中。
        
        Args:
            conditioning: 原始的 conditioning 列表，每个元素形如 [tensor, options]。
            copy_factor (int): 复制因子，指定复制多少份（最小为 0）。
            mode (str): 模式选择，"空白模式"或"本体模式"。
        
        Returns:
            list: 更新后的 conditioning 列表。
        """
        import torch

        if copy_factor == 0:
            # 如果复制因子为 0，则直接返回原始 conditioning
            return (conditioning,)
        
        new_conditioning = []
        for cond in conditioning:
            # cond 结构为 [tensor, options_dict]
            original_tensor = cond[0]
            options = cond[1].copy()

            if mode == "空白模式":
                # 创建与原张量同形状但全部为 0 的新 token
                new_token = torch.zeros_like(original_tensor)
            else:  # mode 为 "本体模式"
                new_token = original_tensor.clone()

            # 根据复制因子复制 copy_factor 份新 token，并沿 token 维度拼接在原 tensor 后面
            replicated = new_token.repeat(1, copy_factor, 1)
            merged_tensor = torch.cat([original_tensor, replicated], dim=1)
            new_conditioning.append([merged_tensor, options])
        
        return (new_conditioning,)

class PaligemmaApplyStyleModel:
    """
    使用 paligemma2 模型提取图像视觉特征，并将其融合到文本条件（conditioning）中。

    输入参数：
        - conditioning: 文本条件列表，每个元素格式为 [tensor, options_dict]
        - style_model: 包含 get_cond 方法的对象，用于将视觉特征转换为条件向量
        - image: 输入图像，支持（图片路径、PIL 图像、numpy 数组或 torch 张量）多种格式
        - model_dir: 字符串，指定视觉模型所在路径或模型标识符，默认值为 "mirror013/paligemma2-vision-model-896"
                     如果 model_dir 是有效的文件夹路径，则直接加载模型；否则调用 snapshot_download 进行下载

    处理流程：
        1. 根据 model_dir 判断是否已经有模型文件夹，如果无则下载至本地
        2. 加载 SiglipImageProcessor 和 SiglipVisionModel，并设置为 evaluation 模式
        3. 根据传入的图像类型转换为 PIL.Image 格式
        4. 利用处理器对图像进行预处理，并前向计算得到视觉特征
        5. 利用 style_model.get_cond 将视觉特征转换为条件向量，flatten 后与原有文本条件拼接
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "image": ("IMAGE",),
                "model_dir": ("STRING", {"default": "mirror013/paligemma2-vision-model-896"})
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "IMAGE")
    FUNCTION = "apply_stylemodel"
    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, conditioning, style_model, image, model_dir):
        import os
        import torch
        from transformers import SiglipVisionModel, SiglipImageProcessor
        from modelscope import snapshot_download

        # 判断 model_dir 是否为有效的文件夹路径
        if os.path.isdir(model_dir):
            vision_model_path = model_dir
        else:
            vision_model_path = snapshot_download(model_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # 加载图像处理器和视觉模型
            processor = SiglipImageProcessor.from_pretrained(vision_model_path)
            vision_model = SiglipVisionModel.from_pretrained(vision_model_path).to(device)
            vision_model.eval()

            # 图像预处理：图像乘以255转换为uint8类型
            image_uint8 = (image * 255).round().clamp(0, 255).to(torch.uint8)
            # 处理器对图像预处理，返回 inputs 对象，其中 inputs.pixel_values 的范围为 [-1,1]，形状为 (B, C, H, W)
            inputs = processor(images=image_uint8, return_tensors="pt").to(device)
            # 前向计算获取视觉特征
            with torch.inference_mode():
                outputs = vision_model(inputs.pixel_values)
        finally:
            if "vision_model" in locals():
                del vision_model
            torch.cuda.empty_cache()

        # 使用 style_model.get_cond 将视觉特征转换为条件向量
        cond = style_model.get_cond(outputs)
        # 将条件向量 flatten 成形状 (1, batch * tokens, hidden_dim)
        cond = cond.flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)

        new_conditioning = []
        for t in conditioning:
            merged = torch.cat((t[0], cond), dim=1)
            new_conditioning.append([merged, t[1].copy()])

        # 将 inputs.pixel_values 转换回正常的图像：
        # 1. 从 [-1, 1] 转换到 [0, 1]
        # 2. 调整维度为 (B, H, W, C)
        # 3. 确保设备与原始 image 相同
        pixel_values = (inputs.pixel_values + 1.0) / 2.0
        pixel_values = pixel_values.permute(0, 2, 3, 1).to(image.device)

        return (new_conditioning, pixel_values)


# 以下字典用于将节点类映射到对应的全局名称，以及在 ComfyUI 中的节点显示名
NEW_NODE_CLASS_MAPPINGS = {
    "NewStyleModelApplySimple": StyleModelApplySimple,
    "NewReduxAdvanced": ReduxAdvanced,
    "NewReduxAdvancedPaligemma": ReduxAdvancedPaligemma,  # 新增的节点类映射
    "DuplicateConditioning": DuplicateConditioning,  # 新增的节点类映射
    "PaligemmaApplyStyleModel": PaligemmaApplyStyleModel  # 新增的节点类映射
}

NEW_NODE_DISPLAY_NAME_MAPPINGS = {
    "New StyleModelApplySimple": "New Apply style model (simple)",
    "New ReduxAdvanced": "New Apply Redux model (advanced)",
    "New ReduxAdvancedPaligemma": "New Apply Redux model (paligemma)",  # 新增的节点显示名
    "Duplicate Conditioning": "Duplicate Conditioning Node",  # 新增的节点显示名
    "Paligemma Apply Style Model": "Paligemma Apply Style Model Node"  # 新增的节点显示名
}


if __name__ == '__main__':
    # 测试 generate_slices 函数的可行性
    import torch

    # 生成不同尺寸测试图
    sizes = [512, 768, 1024, 1536]
    base_size = 384
    overlap_ratio = 0.1

    for size in sizes:
        # 创建一个示例图像张量 (B, H, W, C)
        image_tensor = torch.rand((1, size, size, 3))  # 生成随机RGB图像

        # 调用 generate_slices 函数
        # slices = generate_slices(image_tensor, base_size=base_size, overlap_ratio=overlap_ratio)
        slices = generate_slices_v2(image_tensor, base_size=384)

        # 输出切片信息
        print(f"Image size: {size}x{size}")
        for i, (slice_tensor, coords) in enumerate(slices):
            print(f"  Slice {i}: Shape {slice_tensor.shape}, Coordinates {coords}")

