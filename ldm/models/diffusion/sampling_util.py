import torch
import numpy as np


def append_dims(x, target_dims):
    """
    テンソルの末尾に次元を追加する関数。追加後の次元数がtarget_dimsに達するまで次元を追加します。
    :param x: 入力テンソル
    :param target_dims: 目標とする次元数
    :return: 次元が追加されたテンソル
    """
    # 追加する次元数を計算
    dims_to_append = target_dims - x.ndim
    # 追加する次元数が負の場合はエラー
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    # 指定された次元数になるまで次元を追加
    return x[(...,) + (None,) * dims_to_append]


def norm_thresholding(x0, value):
    """
    ノルムスレッショルディングを適用する関数。テンソルのノルムが特定の値以下にならないように調整します。
    :param x0: 入力テンソル
    :param value: スレッショルドの値
    :return: ノルムが調整されたテンソル
    """
    # テンソルのノルムを計算し、スレッショルド値でクランプ
    s = append_dims(x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value), x0.ndim)
    # 元のテンソルをスケーリング
    return x0 * (value / s)


def spatial_norm_thresholding(x0, value):
    """
    空間的ノルムスレッショルディングを適用する関数。各位置のノルムが特定の値以下にならないように調整します。
    :param x0: 入力テンソル (b, c, h, w)
    :param value: スレッショルドの値
    :return: ノルムが調整されたテンソル
    """
    # 各位置のノルムを計算し、スレッショルド値でクランプ
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    # 元のテンソルをスケーリング
    return x0 * (value / s)