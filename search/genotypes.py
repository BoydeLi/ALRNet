from collections import namedtuple

Genotype = namedtuple('Genotype', 'Attention1_b1 Attention1_b1_concat Attention1_b2 Attention1_b2_concat Attention1_b3 Attention1_b3_concat Attention1_b4 Attention1_b4_concat Attention2_b1 Attention2_b1_concat Attention2_b2 Attention2_b2_concat Attention2_b3 Attention2_b3_concat Attention2_b4 Attention2_b4_concat Attention3_b1 Attention3_b1_concat Attention3_b2 Attention3_b2_concat Attention3_b3 Attention3_b3_concat Attention3_b4 Attention3_b4_concat Attention4_b1 Attention4_b1_concat Attention4_b2 Attention4_b2_concat Attention4_b3 Attention4_b3_concat Attention4_b4 Attention4_b4_concat Attention5_b1 Attention5_b1_concat Attention5_b2 Attention5_b2_concat Attention5_b3 Attention5_b3_concat Attention5_b4 Attention5_b4_concat CoarseAggre_1 CoarseAggre_1_concat CoarseAggre_2 CoarseAggre_2_concat AdjacentAggre AdjacentAggre_concat Refine_seg Refine_seg_concat Refine_edge Refine_edge_concat Refine_agg Refine_agg_concat Low_seg Low_seg_concat Low_edge Low_edge_concat Low_agg Low_agg_concat')

att_PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_3x3_dil4',
    'spatial_attention',
    'channel_attention',
]

PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_3x3_dil4',
    'spatial_attention',
    'channel_attention',
]

DARTS_ALRNet = Genotype(Attention1_b1=[('sep_conv_3x3', 0), ('sep_conv_3x3', 0)], Attention1_b1_concat=range(1, 3), Attention1_b2=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], Attention1_b2_concat=range(1, 3), Attention1_b3=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], Attention1_b3_concat=range(1, 3), Attention1_b4=[('sep_conv_5x5', 0), ('conv_5x5', 0)], Attention1_b4_concat=range(1, 3), Attention2_b1=[('dil_conv_3x3', 0), ('conv_3x3', 0)], Attention2_b1_concat=range(1, 3), Attention2_b2=[('conv_7x7', 0), ('spatial_attention', 1)], Attention2_b2_concat=range(1, 3), Attention2_b3=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], Attention2_b3_concat=range(1, 3), Attention2_b4=[('dil_conv_3x3', 0), ('conv_7x7', 0)], Attention2_b4_concat=range(1, 3), Attention3_b1=[('conv_3x3', 0), ('channel_attention', 1)], Attention3_b1_concat=range(1, 3), Attention3_b2=[('skip_connect', 0), ('dil_conv_3x3_dil4', 1)], Attention3_b2_concat=range(1, 3), Attention3_b3=[('dil_conv_3x3_dil4', 0), ('skip_connect', 1)], Attention3_b3_concat=range(1, 3), Attention3_b4=[('dil_conv_3x3_dil4', 0), ('dil_conv_3x3_dil4', 0)], Attention3_b4_concat=range(1, 3), Attention4_b1=[('dil_conv_3x3', 0), ('conv_3x3', 0)], Attention4_b1_concat=range(1, 3), Attention4_b2=[('dil_conv_3x3_dil4', 0), ('conv_7x7', 0)], Attention4_b2_concat=range(1, 3), Attention4_b3=[('dil_conv_3x3_dil4', 0), ('dil_conv_3x3_dil4', 1)], Attention4_b3_concat=range(1, 3), Attention4_b4=[('skip_connect', 0), ('dil_conv_3x3', 1)], Attention4_b4_concat=range(1, 3), Attention5_b1=[('skip_connect', 0), ('dil_conv_3x3', 1)], Attention5_b1_concat=range(1, 3), Attention5_b2=[('dil_conv_3x3', 0), ('channel_attention', 0)], Attention5_b2_concat=range(1, 3), Attention5_b3=[('dil_conv_3x3_dil4', 0), ('conv_7x7', 1)], Attention5_b3_concat=range(1, 3), Attention5_b4=[('spatial_attention', 0), ('sep_conv_5x5', 1)], Attention5_b4_concat=range(1, 3), CoarseAggre_1=[('conv_3x3', 1), ('sep_conv_5x5', 0), ('conv_7x7', 1), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('conv_7x7', 2)], CoarseAggre_1_concat=range(2, 6), CoarseAggre_2=[('skip_connect', 1), ('channel_attention', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('conv_7x7', 3), ('conv_3x3', 0), ('spatial_attention', 3), ('conv_3x3', 0)], CoarseAggre_2_concat=range(2, 6), AdjacentAggre=[('conv_5x5', 1), ('conv_7x7', 0), ('skip_connect', 2), ('conv_5x5', 0), ('channel_attention', 2), ('channel_attention', 3), ('conv_3x3', 3), ('spatial_attention', 2)], AdjacentAggre_concat=range(2, 6), Refine_seg=[('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('channel_attention', 3), ('conv_5x5', 1), ('sep_conv_3x3', 2), ('conv_3x3', 3), ('sep_conv_3x3', 2), ('conv_3x3', 4), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('dil_conv_3x3', 5)], Refine_seg_concat=range(3, 7), Refine_edge=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('conv_7x7', 2), ('dil_conv_3x3_dil4', 3), ('conv_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 0)], Refine_edge_concat=range(2, 6), Refine_agg=[('conv_5x5', 0), ('channel_attention', 1), ('conv_7x7', 2), ('skip_connect', 1), ('conv_3x3', 2), ('spatial_attention', 3), ('sep_conv_3x3', 2), ('conv_3x3', 3)], Refine_agg_concat=range(2, 6), Low_seg=[('skip_connect', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('conv_3x3', 2), ('sep_conv_3x3', 1), ('conv_7x7', 0), ('sep_conv_3x3', 2), ('conv_7x7', 3), ('dil_conv_3x3_dil4', 4), ('conv_3x3', 3), ('spatial_attention', 2), ('sep_conv_5x5', 5)], Low_seg_concat=range(3, 7), Low_edge=[('sep_conv_5x5', 1), ('channel_attention', 0), ('conv_7x7', 1), ('dil_conv_3x3', 2), ('conv_3x3', 3), ('conv_5x5', 2), ('sep_conv_5x5', 4), ('dil_conv_3x3_dil4', 2)], Low_edge_concat=range(2, 6), Low_agg=[('conv_3x3', 1), ('channel_attention', 0), ('conv_5x5', 0), ('dil_conv_3x3_dil4', 2), ('dil_conv_3x3_dil4', 2), ('sep_conv_3x3', 0), ('conv_7x7', 0), ('sep_conv_3x3', 4)], Low_agg_concat=range(2, 6))
DARTS = DARTS_ALRNet