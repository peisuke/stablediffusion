from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    # xformersライブラリが利用可能かどうかをチェックし、その結果を変数に格納
    XFORMERS_IS_AVAILBLE = True
except:
    # xformersライブラリが利用不可能な場合、変数をFalseに設定
    XFORMERS_IS_AVAILBLE = False

# 環境変数から注意機構（Attention Mechanism）の精度設定を取得
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    """
    与えられた値がNoneではないかどうかをチェックするヘルパー関数。
    :param val: チェックする値
    :return: 値がNoneでなければTrue、そうでなければFalse
    """
    return val is not None

def uniq(arr):
    """
    配列のユニークな要素のみを返すヘルパー関数。
    :param arr: 入力配列
    :return: ユニークな要素のセット
    """
    return{el: True for el in arr}.keys()

def default(val, d):
    """
    与えられた値がNoneでなければその値を、そうでなければデフォルト値を返すヘルパー関数。
    :param val: チェックする値
    :param d: デフォルト値（関数または値）
    :return: valまたはデフォルト値
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    """
    与えられたテンソルのデータ型に対する最大負値を返すヘルパー関数。
    :param t: テンソル
    :return: 最大負値
    """
    return -torch.finfo(t.dtype).max

def init_(tensor):
    """
    与えられたテンソルを特定の範囲で初期化するヘルパー関数。
    :param tensor: 初期化するテンソル
    :return: 初期化されたテンソル
    """
    # テンソルの最後の次元のサイズを取得
    dim = tensor.shape[-1]
    # 標準偏差を計算（次元の逆平方根）
    std = 1 / math.sqrt(dim)
    # テンソルを一様分布で初期化
    tensor.uniform_(-std, std)
    return tensor


# GEGLU活性化関数を持つニューラルネットワークレイヤーのクラス
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        GEGLUレイヤーの初期化を行います。
        :param dim_in: 入力次元
        :param dim_out: 出力次元
        """
        super().__init__()
        # dim_outの2倍の出力を持つ線形層を作成
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        """
        フォワードパス。
        :param x: 入力テンソル
        :return: 出力テンソル
        """
        # projを通した後、チャンクに分割してGEGLU活性化関数を適用
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


# フィードフォワードネットワークのクラス
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        """
        フィードフォワードネットワークの初期化を行います。
        :param dim: 入力次元
        :param dim_out: 出力次元（省略可能、デフォルトは入力次元と同じ）
        :param mult: 内部次元の拡大係数
        :param glu: GEGLUを使用するかどうか
        :param dropout: ドロップアウト率
        """
        super().__init__()
        # 内部次元を計算
        inner_dim = int(dim * mult)
        # 出力次元が指定されていなければ、入力次元と同じに設定
        dim_out = default(dim_out, dim)
        # 入力を内部次元に射影するレイヤーを作成（GEGLUの有無に応じて変更）
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        # ネットワークを構成
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        """
        フォワードパス。
        :param x: 入力テンソル
        :return: 出力テンソル
        """
        return self.net(x)

def zero_module(module):
    """
    モジュールのパラメータをすべてゼロに初期化する関数。
    :param module: 初期化するモジュール
    :return: 初期化されたモジュール
    """
    # モジュールの各パラメータに対して
    for p in module.parameters():
        # パラメータの値をdetachしてゼロに設定
        p.detach().zero_()
    # 初期化されたモジュールを返す
    return module


def Normalize(in_channels):
    """
    指定されたチャンネル数の入力に対してGroupNormを作成する関数。
    :param in_channels: 入力チャンネル数
    :return: GroupNormレイヤー
    """
    # GroupNormレイヤーを作成し、返す
    # ここでは、グループ数を32、チャンネル数を入力チャンネル数に設定
    # epsは数値安定性のために設定された小さな値、affineは学習可能なスケールとシフトパラメータを使用するかどうか
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    """
    空間的自己注意機構（Spatial Self-Attention）を実装するニューラルネットワークのモジュール。
    このモジュールは、入力特徴マップに自己注意を適用し、特徴間の関連性を学習する。
    """

    def __init__(self, in_channels):
        """
        コンストラクタ。
        :param in_channels: 入力チャンネル数
        """
        super().__init__()
        self.in_channels = in_channels

        # 入力特徴マップの正規化を行うノームレイヤー
        self.norm = Normalize(in_channels)
        # クエリ、キー、バリューのための1x1畳み込み層を定義
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 出力のための1x1畳み込み層
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        フォワードパス。
        :param x: 入力テンソル
        :return: 出力テンソル
        """
        # 入力テンソルを正規化
        h_ = self.norm(x)
        # クエリ、キー、バリューを計算
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 注意重みを計算
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        # スケーリング係数で注意重みを調整し、ソフトマックスを適用
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # バリューに注意重みを適用
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        # 出力プロジェクションを適用
        h_ = self.proj_out(h_)

        # 元の入力と加算して出力
        return x + h_


class CrossAttention(nn.Module):
    """
    クロスアテンション（Cross Attention）を実装するニューラルネットワークのモジュール。
    クエリ（query）とコンテキスト（context）の間で注意機構を計算します。
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        """
        コンストラクタ。
        :param query_dim: クエリの次元
        :param context_dim: コンテキストの次元（省略可能）
        :param heads: マルチヘッドアテンションのヘッド数
        :param dim_head: 各ヘッドの次元
        :param dropout: ドロップアウト率
        """
        super().__init__()
        # 内部次元（各ヘッドの次元 × ヘッド数）
        inner_dim = dim_head * heads
        # コンテキストの次元が指定されていなければ、クエリの次元と同じにする
        context_dim = default(context_dim, query_dim)

        # スケーリング係数
        self.scale = dim_head ** -0.5
        self.heads = heads

        # クエリ、キー、バリュー用の線形層
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 出力用の線形層とドロップアウト
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        フォワードパス。
        :param x: 入力テンソル（クエリ）
        :param context: コンテキストテンソル（省略可能）
        :param mask: アテンションマスク（省略可能）
        :return: アテンションの結果
        """
        h = self.heads

        # クエリ、キー、バリューを計算
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # マルチヘッドアテンションの形式に変換
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # アテンションスコアを計算
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        # クエリとキーはもはや不要なので削除
        del q, k
    
        # マスクが存在する場合、アテンションスコアに適用
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # ソフトマックスを適用して正規化
        sim = sim.softmax(dim=-1)

        # バリューにアテンションスコアを適用し、出力を計算
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    """
    メモリ効率の良いクロスアテンションを実装するニューラルネットワークのモジュール。
    xformersライブラリのmemory_efficient_attentionを使用しています。
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        """
        コンストラクタ。
        :param query_dim: クエリの次元
        :param context_dim: コンテキストの次元（省略可能、デフォルトはquery_dimと同じ）
        :param heads: マルチヘッドアテンションのヘッド数
        :param dim_head: 各ヘッドの次元
        :param dropout: ドロップアウト率
        """
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        # 内部次元（各ヘッドの次元 × ヘッド数）
        inner_dim = dim_head * heads
        # コンテキストの次元が指定されていなければ、クエリの次元と同じにする
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        # クエリ、キー、バリュー用の線形層
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 出力用の線形層とドロップアウト
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 注意演算のためのオプションを保持
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        """
        フォワードパス。
        :param x: 入力テンソル（クエリ）
        :param context: コンテキストテンソル（省略可能）
        :param mask: アテンションマスク（省略可能、未実装）
        :return: アテンションの結果
        """
        # クエリ、キー、バリューを計算
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        # マルチヘッドアテンションの形式に変換
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # メモリ効率の良いアテンション演算を実行
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # マスクが存在する場合、エラー（未実装）
        if exists(mask):
            raise NotImplementedError
        # 出力を元の形式に再整形
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    ベーシックなトランスフォーマーブロックを実装するニューラルネットワークのモジュール。
    2つのアテンションレイヤと1つのフィードフォワードレイヤを含む。
    """

    # 利用可能なアテンションモードを定義
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 通常のアテンション
        "softmax-xformers": MemoryEfficientCrossAttention  # メモリ効率の良いアテンション（xformers利用時）
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        """
        コンストラクタ。
        :param dim: 入力次元
        :param n_heads: ヘッド数
        :param d_head: 各ヘッドの次元
        :param dropout: ドロップアウト率
        :param context_dim: コンテキスト次元（省略可能）
        :param gated_ff: ゲーテッドフィードフォワードを使用するかどうか
        :param checkpoint: チェックポイントを使用するかどうか
        :param disable_self_attn: 自己アテンションを無効化するかどうか
        """
        super().__init__()
        # xformersが利用可能かどうかに基づいてアテンションモードを選択
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        # 第一のアテンションレイヤ（自己アテンションまたはクロスアテンション）
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)
        # フィードフォワードレイヤ
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 第二のアテンションレイヤ（自己アテンションまたはクロスアテンション）
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)
        # 正規化レイヤ
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        """
        フォワードパス。
        :param x: 入力テンソル
        :param context: コンテキストテンソル（省略可能）
        :return: 出力テンソル
        """
        # チェックポイントを利用して_forward関数を実行
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        """
        実際のフォワードパス。
        :param x: 入力テンソル
        :param context: コンテキストテンソル（省略可能）
        :return: 出力テンソル
        """
        # 第一のアテンションレイヤを適用
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        # 第二のアテンションレイヤを適用
        x = self.attn2(self.norm2(x), context=context) + x
        # フィードフォワードレイヤを適用
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    画像データのためのトランスフォーマーブロック。
    入力をまずプロジェクト（埋め込み）してから、b, t, dの形状に変形し、
    標準的なトランスフォーマー処理を適用します。
    最後に、画像の形状に戻します。
    新しいオプション：use_linearを使用して、1x1畳み込みの代わりに効率的に処理
    """

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        """
        コンストラクタ。
        :param in_channels: 入力チャンネル数
        :param n_heads: ヘッド数
        :param d_head: 各ヘッドの次元
        :param depth: トランスフォーマーブロックの深さ
        :param dropout: ドロップアウト率
        :param context_dim: コンテキスト次元（省略可能、リストで指定可能）
        :param disable_self_attn: 自己アテンションを無効化するかどうか
        :param use_linear: 線形層を使用するかどうか
        :param use_checkpoint: チェックポイントを使用するかどうか
        """
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        # トランスフォーマーブロックのリストを作成
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        """
        フォワードパス。
        :param x: 入力テンソル
        :param context: コンテキストテンソル（省略可能、リストで指定可能）
        :return: 出力テンソル
        """
        # コンテキストが指定されていない場合は、自己アテンションとして処理
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in