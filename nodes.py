import numpy as np
import torch
import comfy
import folder_paths
import nodes
import os
import math
import re
import safetensors
import glob
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
    在许多图像处理场景里，如果直接对图像进行裁剪或缩放，有时会破坏图像原本的长宽比。为保持图像纵横比不变，或填充成特定尺寸的方形输出，我们常用到一种被称为“letterbox”的处理方式。
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
    "autocrop with mask"
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
                             "weight": ("FLOAT", {"default": 1.0, "min":0.0, "max":1.0, "step":0.01})
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
                         mask=None, autocrop_margin=0.0):
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
            mode (str): "center crop (square)" / "keep aspect ratio" / "autocrop with mask"。
            weight (float): 融合权重，越大则该图像特征在最终生成中的影响力越大。
            mask (torch.FloatTensor or None): 掩膜，用于指定关注的图像区域。
            autocrop_margin (float): 在 "autocrop with mask" 时，给前景区域额外留出的边距。
        
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
        image, masko = prepareImageAndMask(clip_vision, image, mask, mode, autocrop_margin)
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
        mode="area"  # 此处可视为固定最终 mask 插值方式；实际插值方式受 downsampling_function 控制

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
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        """
        cond 张量维度是torch.Size([1, 729, 4096]),其中4096是文本编码器的维度，而729则是siglip模型的将图像分成27*27=729个小块
        """
        (b,t,h) = cond.shape

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
                mode=mode
            ).transpose(-1,1)
        
        #  将条件向量 cond 乘以权重平方，以调整其在最终条件中的影响力度
        # 不直接使用线性权重（cond * weight），而是使用平方权重（cond * weight * weight），
        # 这样可以确保权重在最终条件中的影响力度是线性的，而不是平方的。
        cond = cond * (weight * weight)
        
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

# 以下字典用于将节点类映射到对应的全局名称，以及在 ComfyUI 中的节点显示名
NODE_CLASS_MAPPINGS = {
    "StyleModelApplySimple": StyleModelApplySimple,
    "ReduxAdvanced": ReduxAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelApplySimple": "Apply style model (simple)",
    "ReduxAdvanced": "Apply Redux model (advanced)"
}