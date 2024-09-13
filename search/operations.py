import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_3x3_dil3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 3, 3, affine=affine),
  'dil_conv_3x3_dil5' : lambda C, stride, affine: DilConv(C, C, 3, stride, 5, 5, affine=affine),
  'dil_conv_3x3_dil7' : lambda C, stride, affine: DilConv(C, C, 3, stride, 7, 7, affine=affine),
  'dil_conv_3x3_dil4' : lambda C, stride, affine: DilConv(C, C, 3, stride, 4, 4, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_3x3' : lambda C, stride, affine : ReLUConvBN(C, C, 3, stride, 1, affine=affine),
  'conv_5x5' : lambda C, stride, affine : ConvDouble(C, C, 3, stride, 1, affine=affine),
  'conv_7x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'spatial_attention': lambda C, stride, affine : SpatialAttention(C, 7),
  'channel_attention': lambda C, stride, affine : ChannelAttention(C, 16),
  'self_attention': lambda C, stride, affine : SelfAttention(C, 1, 1)
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class ConvDouble(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ConvDouble, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module): #for skip_connect in reduction cell

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


class SpatialAttention(nn.Module):
  def __init__(self, in_C, kernel_size=7):
    super(SpatialAttention, self).__init__()

    assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
    padding = 3 if kernel_size == 7 else 1
    self.in_channels = in_C
    self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    self.sigmoid = nn.Sigmoid()
    self.conv11 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
    self.bn11 = nn.BatchNorm2d(self.in_channels)

  def forward(self, x):
    input = x
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.conv1(x)
    x = self.sigmoid(x)
    out = input * x

    out = F.relu(self.bn11(self.conv11(out)), inplace=True)

    return out

class ChannelAttention(nn.Module):
  def __init__(self, in_channels, ratio=16):
    super(ChannelAttention, self).__init__()

    self.in_channels = in_channels

    self.linear_1 = nn.Linear(self.in_channels, self.in_channels // ratio)
    self.linear_2 = nn.Linear(self.in_channels // ratio, self.in_channels)
    self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(self.in_channels)

  def forward(self, input_):
    n_b, n_c, h, w = input_.size()

    feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
    feats = F.relu(self.linear_1(feats))
    feats = torch.sigmoid(self.linear_2(feats))

    feats = feats.view((n_b, n_c, 1, 1))
    feats = feats.expand_as(input_).clone()
    out = torch.mul(input_, feats)
    out = F.relu(self.bn1(self.conv1(out)), inplace=True)

    return out

class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class Attention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads

    self.scale = qk_scale or head_dim ** -0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class TransBlock(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = Attention(
      dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

  def forward(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

class SelfAttention(nn.Module):
  def __init__(self, embed_dim, depth, num_heads):
    super(SelfAttention, self).__init__()
    self.token_trans = TransBlock(embed_dim, num_heads, mlp_ratio=3.)
    for m in self.modules():
      classname = m.__class__.__name__
      if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight),
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight),
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    feat_t = x
    B, Ct, Ht, Wt = feat_t.shape
    feat_t = feat_t.view(B, Ct, -1).transpose(1, 2)
    Tt = self.token_trans(feat_t)
    mask_x = Tt.transpose(1, 2).reshape(B, Ct, Ht, Wt)
    return mask_x