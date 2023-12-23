"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    """
    ガウス拡散を用いたクラシックなDDPM（Denoising Diffusion Probabilistic Models）の実装。
    画像空間での拡散と復元プロセスを扱います。
    """

    def __init__(self, unet_config, timesteps=1000, beta_schedule="linear", loss_type="l2",
                 ckpt_path=None, ignore_keys=[], load_only_unet=False, monitor="val/loss",
                 use_ema=True, first_stage_key="image", image_size=256, channels=3, log_every_t=100,
                 clip_denoised=True, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3,
                 given_betas=None, original_elbo_weight=0., v_posterior=0.,
                 l_simple_weight=1., conditioning_key=None, parameterization="eps",
                 scheduler_config=None, use_positional_encodings=False, learn_logvar=False,
                 logvar_init=0., make_it_fit=False, ucg_training=None, reset_ema=False,
                 reset_num_ema_updates=False):
        """
        コンストラクタ。
        :param unet_config: U-Netモデルの設定
        :param timesteps: 拡散プロセスのタイムステップ数
        :param beta_schedule: βのスケジューリング方法（"linear", "cosine"など）
        :param loss_type: 損失関数の種類（"l2", "l1"など）
        :param ckpt_path: モデルのチェックポイントパス
        :param ignore_keys: チェックポイントから無視するキー
        :param load_only_unet: U-Netのみをロードするかどうか
        :param monitor: 監視する値（例：'val/loss'）
        :param use_ema: 指数移動平均（EMA）を使用するかどうか
        :param first_stage_key: 最初のステージのキー
        :param image_size: 画像のサイズ
        :param channels: チャンネル数
        :param log_every_t: ログを取るタイミング
        :param clip_denoised: 復元された画像をクリップするかどうか
        :param linear_start: 線形スケジューリングの開始値
        :param linear_end: 線形スケジューリングの終了値
        :param cosine_s: コサインスケジューリングのパラメータ
        :param given_betas: 与えられたβの値
        :param original_elbo_weight: ELBOの重み
        :param v_posterior: 後続分布の分散の重み
        :param l_simple_weight: 簡単な損失の重み
        :param conditioning_key: 条件付けのキー
        :param parameterization: パラメータ化の方法（"eps", "x0", "v"）
        :param scheduler_config: スケジューラの設定
        :param use_positional_encodings: 位置エンコーディングを使用するかどうか
        :param learn_logvar: 対数分散を学習するかどうか
        :param logvar_init: 対数分散の初期値
        :param make_it_fit: メモリにフィットさせるためのフラグ
        :param ucg_training: UCGトレーニングの設定
        :param reset_ema: EMAをリセットするかどうか
        :param reset_num_ema_updates: EMAの更新回数をリセットするかどうか
        """
        super().__init__()
        # パラメータ化の方法をチェック
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        # 条件付けステージモデルの初期化
        self.cond_stage_model = None
        # 各種設定の初期化
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        # DiffusionWrapperモデルの初期化
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        # EMAの設定
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        # スケジューラの設定
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        # 各種重みの設定
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        # モニターの設定
        if monitor is not None:
            self.monitor = monitor
        self.make_it_fit = make_it_fit
        # チェックポイントとEMAのリセット
        if reset_ema: assert exists(ckpt_path)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            if reset_ema:
                assert self.use_ema
                print(f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()
        # 拡散スケジュールの登録
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # 損失タイプの設定
        self.loss_type = loss_type
        # 対数分散の設定
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        # UCGトレーニングの設定
        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """
        拡散プロセスのスケジュールを登録する関数。
        :param given_betas: 与えられたベータの配列（省略可能）
        :param beta_schedule: ベータのスケジューリング方法（'linear', 'cosine'など）
        :param timesteps: 拡散プロセスのタイムステップ数
        :param linear_start: 線形スケジュールの開始値
        :param linear_end: 線形スケジュールの終了値
        :param cosine_s: コサインスケジュールのパラメータ
        """
        # 与えられたベータがある場合はそれを使用し、なければスケジュールに基づいて生成
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 拡散スケジュールに関連するパラメータをバッファとして登録
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # 拡散q(x_t | x_{t-1})に関連する計算
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # 後続分布q(x_{t-1} | x_t, x_0)に関連する計算
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # LVLB重みの計算（損失関数に関連）
        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        """
        EMA（Exponential Moving Average）重みを一時的に適用するためのコンテキストマネージャー。
        :param context: コンテキスト（オプション、デバッグやロギングに使用）
        """
        # EMAを使用する場合
        if self.use_ema:
            # 現在のモデルのパラメータを保存
            self.model_ema.store(self.model.parameters())
            # EMA重みをモデルにコピー
            self.model_ema.copy_to(self.model)
            # コンテキストが指定されている場合、コンソールにメッセージを表示
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            # コンテキストマネージャー内の処理を実行
            yield None
        finally:
            # EMAを使用する場合
            if self.use_ema:
                # 保存しておいた元のモデルのパラメータを復元
                self.model_ema.restore(self.model.parameters())
                # コンテキストが指定されている場合、コンソールにメッセージを表示
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        """
        チェックポイントからモデルの重みを初期化するメソッド。指定されたパスからチェックポイントをロードし、モデルの重みを初期化します。

        :param path: チェックポイントのファイルパス
        :param ignore_keys: 無視するキーのリスト（オプション）
        :param only_model: Trueの場合、モデルのみを初期化（オプション）
        """
        # チェックポイントをCPU上にロード
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        # チェックポイントから読み込むキーのリストを取得
        keys = list(sd.keys())

        # 無視するキーが指定されている場合、該当するキーを削除
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        # make_it_fitオプションが有効な場合、新しいモデルの重みに古い重みを合わせる
        if self.make_it_fit:
            n_params = len([name for name in itertools.chain(self.named_parameters(), self.named_buffers())])
            for name, param in tqdm(
                itertools.chain(self.named_parameters(), self.named_buffers()),
                desc="Fitting old weights to new weights",
                total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # 最初の2つの軸のみを変更する
                    assert new_shape[2:] == old_shape[2:]
                # 最初の軸が出力次元に対応していると仮定
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        # モデルの重みをロード（only_modelがTrueの場合はモデルのみ）
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """

        # この式は、時間ステップtにおける平均を計算します。x_startはノイズがない初期の入力画像です。
        # self.sqrt_alphas_cumprodは、各拡散ステップにおける平方根の累積積です。
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)

        # この式は、時間ステップtにおける分散を計算します。
        # self.alphas_cumprodは、各拡散ステップにおける累積積です。
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)

        # この式は、時間ステップtにおける分散の対数を計算します。
        # self.log_one_minus_alphas_cumprodは、1から各アルファの累積積を引いた値の対数です。
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        ノイズが加えられた画像から、ノイズがない元の画像を推測する。
        :param x_t: ノイズが加えられた画像（テンソル形式）。
        :param t: 拡散のステップ数。
        :param noise: 加えられたノイズ。
        :return: 推測されたノイズがない元の画像。
        """
        # 拡散の逆プロセスにおいて、ノイズがない元の画像を推定します。
        # self.sqrt_recip_alphas_cumprodは、拡散ステップにおけるアルファの累積積の平方根の逆数です。
        # self.sqrt_recipm1_alphas_cumprodは、アルファの累積積の1からの差の平方根の逆数です。
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        """
        ノイズ除去された画像と潜在変数から、ノイズがない元の画像を推測する。
        :param x_t: 拡散後の画像（テンソル形式）。
        :param t: 拡散のステップ数。
        :param v: 潜在変数。
        :return: 推測されたノイズがない元の画像。
        """
        # self.sqrt_alphas_cumprodは、拡散ステップにおけるアルファの累積積の平方根です。
        # self.sqrt_one_minus_alphas_cumprodは、1からアルファの累積積を引いた値の平方根です。
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def predict_eps_from_z_and_v(self, x_t, t, v):
        """
        拡散後の画像と潜在変数から、加えられたノイズを推測する。
        :param x_t: 拡散後の画像（テンソル形式）。
        :param t: 拡散のステップ数。
        :param v: 潜在変数。
        :return: 推測されたノイズ。
        """
        # この関数は、拡散プロセス中に加えられたノイズを推測するために使用されます。
        # self.sqrt_alphas_cumprodとself.sqrt_one_minus_alphas_cumprodは上述の通りです。
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        """
        拡散プロセス中の特定のステップにおける事後分布を計算する。
        :param x_start: ノイズがない元の画像。
        :param x_t: 拡散プロセス中の特定のステップにおける画像。
        :param t: 拡散のステップ数。
        :return: 事後分布の平均、分散、および分散の対数。
        """
        # 事後分布の平均を計算します。この計算は、元の画像と拡散プロセス中の画像に基づいています。
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
    
        # 事後分布の分散を計算します。
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
    
        # 事後分布の分散の対数を計算します。分散の対数は、数値的安定性のためにクリップされることがあります。
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool):
        """
        モデルによる出力から、再構築された画像の推定値と、事後分布のパラメータを計算する。
        :param x: 拡散プロセス中の特定のステップにおける画像。
        :param t: 拡散のステップ数。
        :param clip_denoised: 再構築された画像をクリップするかどうかのフラグ。
        :return: モデルによる推定値、事後分布の分散、事後分布の分散の対数。
        """
        model_out = self.model(x, t)
    
        # モデルの出力に基づいて、ノイズ除去された画像を再構築します。
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
    
        # 再構築された画像が指定された範囲内に収まるようにクリップします。
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
    
        # 再構築された画像を使用して、事後分布のパラメータを計算します。
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        """
        逆拡散プロセスにおいて、1ステップ分のサンプリングを行う。
        :param x: 現在の拡散画像。
        :param t: 拡散のステップ数。
        :param clip_denoised: 再構築された画像をクリップするかどうかのフラグ。
        :param repeat_noise: 同じノイズを繰り返すかどうかのフラグ。
        :return: 次のステップにおける画像。
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)

        # 画像に適用するノイズを生成します。
        noise = noise_like(x.shape, device, repeat_noise)

        # t == 0の時はノイズを加えません。
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        """
        逆拡散プロセス全体を通して画像をサンプリングする。
        :param shape: 生成する画像の形状。
        :param return_intermediates: 中間画像を返すかどうかのフラグ。
        :return: 生成された画像、（オプションで）中間画像のリスト。
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]

        # 逆拡散プロセスを順次実行していきます。
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)

            # 指定された間隔で中間結果を記録します。
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        # 中間画像も返す場合は、最終画像と共に返します。
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        """
        画像を生成するためのメイン関数。バッチサイズと中間結果の返却設定に基づいて、画像をサンプリングする。
        :param batch_size: 生成する画像のバッチサイズ。
        :param return_intermediates: 中間画像を返すかどうかのフラグ。
        :return: サンプリングされた画像、オプションで中間画像のリスト。
        """
        image_size = self.image_size
        channels = self.channels
        # p_sample_loop関数を用いて、画像をサンプリングします。
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        """
        拡散プロセスをシミュレートするための関数。元の画像とノイズを組み合わせて、特定の拡散ステップでの画像を生成する。
        :param x_start: ノイズがない元の画像。
        :param t: 拡散のステップ数。
        :param noise: 追加するノイズ（オプション）。指定されていない場合はランダムノイズが生成される。
        :return: 拡散プロセスによって生成された画像。
        """
        # ノイズが指定されていない場合はランダムノイズを生成します。
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 拡散プロセスをシミュレートします。ここでの計算は、元の画像とノイズを適切な比率で混合することによって行われます。
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def get_v(self, x, noise, t):
        """
        拡散プロセス中の特定のステップでの潜在変数vを計算する。
        :param x: 拡散プロセス中の特定のステップにおける画像。
        :param noise: 加えられたノイズ。
        :param t: 拡散のステップ数。
        :return: 計算された潜在変数v。
        """
        # 拡散プロセス中に加えられたノイズと、そのステップでの画像から潜在変数vを計算します。
        # self.sqrt_alphas_cumprodは、拡散ステップにおけるアルファの累積積の平方根です。
        # self.sqrt_one_minus_alphas_cumprodは、1からアルファの累積積を引いた値の平方根です。
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        """
        モデルの予測とターゲットとの間の損失を計算する。
        :param pred: モデルによる予測値。
        :param target: 実際の目標値。
        :param mean: 損失の平均を取るかどうかのフラグ。
        :return: 計算された損失。
        """
        # L1損失（絶対値損失）を使用する場合
        if self.loss_type == 'l1':
            # 予測とターゲットとの差の絶対値を取る
            loss = (target - pred).abs()

            # 損失の平均を取る場合
            if mean:
                loss = loss.mean()

        # L2損失（平均二乗誤差）を使用する場合
        elif self.loss_type == 'l2':
            # 損失の平均を取る場合
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                # 平均を取らず、各要素の損失を保持する
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')

        # 未知の損失タイプが指定された場合は例外を投げる
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        """
        モデルの損失を計算する。モデルの出力とターゲットに基づいて、様々な損失を計算する。
        :param x_start: ノイズがない元の画像。
        :param t: 拡散のステップ数。
        :param noise: 加えられたノイズ（オプション）。指定されていない場合はランダムノイズが生成される。
        :return: 計算された損失と、損失に関する情報を含む辞書。
        """
        # ノイズが指定されていない場合はランダムノイズを生成する。
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 拡散プロセスに基づいて、ノイズが加えられた画像を生成する。
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # モデルを用いて、ノイズが加えられた画像から出力を生成する。
        model_out = self.model(x_noisy, t)

        loss_dict = {}

        # モデルのパラメータ化に基づいて、ターゲットを設定する。
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            # サポートされていないパラメータ化が指定された場合は例外を投げる。
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        # 損失を計算する。
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        # トレーニング中か検証中かに基づいてログのプレフィックスを設定する。
        log_prefix = 'train' if self.training else 'val'

        # 単純な損失を計算し、ログに記録する。
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        # 変分下限(Variational Lower Bound)の損失を計算し、ログに記録する。
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        # 総損失を計算する。
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        # 総損失を辞書に追加する。
        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        """
        モデルのフォワードパス。入力データを受け取り、損失計算関数を呼び出して損失を計算する。
        :param x: 入力データ。
        :param args: 追加の位置引数。
        :param kwargs: 追加のキーワード引数。
        :return: 損失計算関数からの出力。
        """
        # ランダムに拡散ステップ数を選択する。
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # 損失計算関数を呼び出す。
        return self.p_losses(x, t, *args, **kwargs)
    
    def get_input(self, batch, k):
        """
        バッチから入力データを取得し、適切な形式に変換する。
        :param batch: データバッチ。
        :param k: バッチ内のキー。
        :return: 処理された入力データ。
        """
        # バッチから入力データを取得する。
        x = batch[k]
        # 入力が3次元の場合（例えば、チャネルが欠けている場合）、チャネル次元を追加する。
        if len(x.shape) == 3:
            x = x[..., None]
        # データを適切な形式に変換する（バッチ、チャネル、高さ、幅）。
        x = rearrange(x, 'b h w c -> b c h w')
        # データを適切なメモリ形式に変換し、float型にキャストする。
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def shared_step(self, batch):
        """
        トレーニングまたは検証ステップで共通に行われる処理。入力データを取得し、モデルに渡して損失を計算する。
        :param batch: データバッチ。
        :return: 計算された損失と、損失に関する情報を含む辞書。
        """
        # 入力データを取得する。
        x = self.get_input(batch, self.first_stage_key)
        # モデルのフォワードパスを実行し、損失と損失辞書を取得する。
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        """
        トレーニングの各ステップで実行される処理。
        バッチデータの前処理、損失の計算、ログの記録を行う。
        :param batch: トレーニングバッチ。
        :param batch_idx: バッチのインデックス。
        :return: 計算された損失。
        """
        # データの前処理：特定の条件に基づいて、バッチ内のデータを変更する。
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            # valがNoneの場合、空の文字列に置き換える。
            if val is None:
                val = ""
            # バッチ内の各要素に対して、確率pに基づいて、値をvalに置き換える。
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        # 共通のステップを実行して損失を計算する。
        loss, loss_dict = self.shared_step(batch)

        # 損失に関する情報をログに記録する。
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        # グローバルステップ数をログに記録する。
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # 学習率スケジューラを使用している場合は、現在の学習率をログに記録する。
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        バリデーションステップの処理。バッチデータに対して損失を計算し、EMA（指数移動平均）と非EMAの両方の損失をログに記録する。
        :param batch: バリデーションバッチ。
        :param batch_idx: バッチのインデックス。
        """
        # EMAを使用しない場合の損失計算
        _, loss_dict_no_ema = self.shared_step(batch)

        # EMAを使用した場合の損失計算
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            # EMA損失のキーに '_ema' を追加
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}

        # 非EMAおよびEMAの損失をログに記録
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        """
        トレーニングバッチが終了するたびに呼び出される処理。EMA（指数移動平均）の更新を行う。
        """
        # EMAを使用する場合、モデルのEMAを更新
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        """
        画像サンプルのリストを行形式で整理する。画像グリッドを生成する。
        :param samples: 画像サンプルのリスト。
        :return: 整理された画像グリッド。
        """
        n_imgs_per_row = len(samples)
        # サンプルを行形式に再配置
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        # n_imgs_per_rowの数に基づいて画像グリッドを生成
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        """
        バッチから画像を取得し、拡散過程とデノイズ過程のサンプルをログとして生成する。
        :param batch: データバッチ。
        :param N: ログに記録する画像の最大数。
        :param n_row: 拡散過程の画像を表示する行数。
        :param sample: デノイズ過程のサンプルを生成するかどうか。
        :param return_keys: 返却するログのキー。
        :param kwargs: 追加のキーワード引数。
        :return: 生成されたログの辞書。
        """
        log = dict()
        # 入力データを取得し、ログに記録する。
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # 拡散過程の行を生成する。
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            """
            拡散プロセスの各ステップを順にサンプリングするループ。
            拡散はnum_timestepsステップにわたって進行する。
            """
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                """
                拡散プロセスの特定のステップをログとして記録する。
                log_every_tごとのステップ、または最後のステップの場合に処理を行う。
                """
                # 拡散ステップの数値tをテンソルに変換し、それをバッチサイズ（n_row）の数だけ繰り返す。
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                # テンソルを適切なデバイス（CPUまたはGPU）に移動し、整数型に変換。
                t = t.to(self.device).long()

                # 入力画像x_startに対してランダムノイズを生成。
                noise = torch.randn_like(x_start)

                # 拡散プロセスをシミュレートして、ノイズが加えられた画像を生成。
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

                # 生成されたノイズが加えられた画像を拡散行リストに追加。
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        # サンプリングが有効な場合、デノイズ過程のサンプルを生成する。
        if sample:
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        # 特定のキーに基づいてログを返却する。
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        """
        モデルのオプティマイザーを設定する。
        ここで設定されたオプティマイザーは、モデルのトレーニングに使用される。
        :return: 設定されたオプティマイザー。
        """
        # 学習率を取得する。
        lr = self.learning_rate

        # モデルのパラメータを取得する。
        params = list(self.model.parameters())

        # learn_logvarフラグがTrueの場合、logvarパラメータもオプティマイザーに含める。
        if self.learn_logvar:
            params = params + [self.logvar]

        # オプティマイザーとしてAdamWを使用し、取得したパラメータと学習率で初期化する。
        opt = torch.optim.AdamW(params, lr=lr)

        # 設定されたオプティマイザーを返す。
        return opt


class LatentDiffusion(DDPM):
    """
    DDPMの拡張クラスであり、追加の機能と設定を提供する。
    主に条件付き生成タスクに使用される。
    """

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):
        """
        コンストラクタ。クラスの初期化時に呼び出される。
        :param first_stage_config: 最初のステージの設定。
        :param cond_stage_config: 条件付けステージの設定。
        :param num_timesteps_cond: 条件付けに用いるタイムステップの数。
        :param cond_stage_key: 条件付けステージのキー。
        :param cond_stage_trainable: 条件付けステージがトレーニング可能かどうか。
        :param concat_mode: 連結モードを使用するかどうか。
        :param cond_stage_forward: 条件付けステージのフォワード関数。
        :param conditioning_key: 条件付けのキー。
        :param scale_factor: スケールファクター。
        :param scale_by_std: 標準偏差によるスケーリングを使用するかどうか。
        :param force_null_conditioning: null条件付けを強制するかどうか。
        :param args: 追加の位置引数。
        :param kwargs: 追加のキーワード引数。
        """
        self.force_null_conditioning = force_null_conditioning
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # 互換性を保つための条件付けキーの設定
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__' and not self.force_null_conditioning:
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        # 最初のステージの構成を取得
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        # 最初のステージと条件付けステージの初期化
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        # チェックポイントからの再開処理
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            if reset_ema:
                assert self.use_ema
                print("EMAを純粋なモデルの重みにリセットします。EMAのみのチェックポイントから復元する場合に便利です。")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print("警告: NUM_EMA UPDATESをゼロにリセットします。")
            assert self.use_ema
            self.model_ema.reset_num_updates()
    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """
        トレーニングの各バッチの開始時に呼び出されるメソッド。
        特に、最初のエポックの最初のバッチで標準偏差によるリスケーリングを行う。
        :param batch: 現在のバッチデータ。
        :param batch_idx: 現在のバッチのインデックス。
        :param dataloader_idx: 使用されているデータローダーのインデックス。
        """
        # 標準偏差によるスケーリングが有効で、かつ最初のエポックの最初のバッチである場合に処理を行う
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            # カスタムリスケーリングと標準偏差リスケーリングを同時に使用しないようアサート
            assert self.scale_factor == 1., 'カスタムリスケーリングと標準偏差リスケーリングを同時に使用するのは推奨されません'

            # リスケールの重みをエンコーディングの標準偏差の逆数に設定
            print("### 標準偏差リスケーリングを使用します ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

            # scale_factorを削除し、新しい値で登録
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"self.scale_factorを{self.scale_factor}に設定しました")
            print("### 標準偏差リスケーリングを使用します ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """
        拡散プロセスのスケジュールを登録するメソッド。
        :param given_betas: 事前に定義されたベータ値のリスト（オプション）。
        :param beta_schedule: ベータ値のスケジュールタイプ（"linear"または"cosine"）。
        :param timesteps: 拡散プロセスの総ステップ数。
        :param linear_start: 線形スケジュールの開始値。
        :param linear_end: 線形スケジュールの終了値。
        :param cosine_s: コサインスケジュールのs値。
        """
        # 親クラスのregister_scheduleメソッドを呼び出し、ベータ値のスケジュールを登録する。
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        # 条件付けスケジュールを短縮するかどうかを判断するフラグを設定する。
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            # 条件付けスケジュールを作成する。
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        """
        最初のステージのモデルをインスタンス化するメソッド。
        :param config: モデルの設定。
        """
        # 設定に基づいてモデルをインスタンス化する。
        model = instantiate_from_config(config)

        # モデルを評価モードに設定し、トレーニングを無効にする。
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train

        # モデルの全パラメータの勾配計算を無効にする。
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        """
        条件付けステージのモデルをインスタンス化するメソッド。
        :param config: 条件付けステージの設定。
        """
        # 条件付けステージがトレーニング可能でない場合
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                # 最初のステージのモデルを条件付けステージとして使用する。
                print("最初のステージのモデルを条件付けステージとして使用します。")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                # 条件付けを行わないモデルとしてトレーニングする。
                print(f"{self.__class__.__name__} を条件付けなしのモデルとしてトレーニングします。")
                self.cond_stage_model = None
            else:
                # 設定に基づいて条件付けステージのモデルをインスタンス化し、評価モードに設定する。
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                # モデルの全パラメータの勾配計算を無効にする。
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            # 条件付けステージがトレーニング可能である場合
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            # 設定に基づいて条件付けステージのモデルをインスタンス化する。
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        """
        リストからノイズ除去された画像の行を取得するメソッドです。
        :param samples: ノイズ除去されたサンプル画像のリスト。
        :param desc: 進行状況の説明（オプション）。
        :param force_no_decoder_quantization: デコーダの量子化を強制しないフラグ（オプション）。
        :return: ノイズ除去された画像の行。
        """
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            # 最初のステージのデコードを使用してノイズ除去を行い、結果をリストに追加します。
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        """
        最初のステージのエンコーディングを取得するメソッドです。
        :param encoder_posterior: エンコーダの事後分布。
        :return: スケールファクターを適用したエンコーディング。
        """
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            # エンコーダの事後分布が対角ガウス分布の場合、サンプリングを行います。
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            # エンコーダの事後分布がテンソルの場合、そのまま使用します。
            z = encoder_posterior
        else:
            # その他の場合は未実装のエラーを発生させます。
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        """
        学習済みのコンディショニングを取得するメソッド。
        このメソッドは、与えられたコンディショニング情報を処理し、
        条件付けステージのモデルを通して、適切な形式に変換します。
        :param c: コンディショニング情報。
        :return: 変換後の学習済みのコンディショニング。
        """
        # 条件付けステージのフォワード関数が定義されていない場合
        if self.cond_stage_forward is None:
            # 条件付けステージのモデルにencodeメソッドがあり、それが呼び出し可能であれば、そのメソッドを使用する。
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                # エンコードされた結果がDiagonalGaussianDistribution型であれば、そのモード（平均）を使用する。
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                # encodeメソッドがない場合は、モデルに直接コンディショニング情報を渡す。
                c = self.cond_stage_model(c)
        else:
            # 条件付けステージのフォワード関数が定義されている場合、その関数を使用する。
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        """
        2次元のメッシュグリッドを作成するメソッド。
        このメソッドは、指定された高さ(h)と幅(w)に基づいて、各点の座標を含むグリッドを生成します。
        :param h: グリッドの高さ。
        :param w: グリッドの幅。
        :return: 作成されたメッシュグリッド。各要素は[y, x]形式の座標。
        """
        # 高さ方向の座標を生成し、適切な形状に変形する。
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        # 幅方向の座標を生成し、適切な形状に変形する。
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        # 高さ方向と幅方向の座標を結合し、最終的なメッシュグリッドを形成する。
        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        画像の各ピクセルの境界までの距離を計算するメソッド。
        距離は正規化され、境界で最小値0、画像中心で最大値0.5をとる。
        :param h: 画像の高さ。
        :param w: 画像の幅。
        :return: 画像の各ピクセルから最も近い境界までの正規化された距離。
        """
        # 右下のコーナーの座標を計算。
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)

        # メッシュグリッドを生成し、右下のコーナーの座標で正規化。
        arr = self.meshgrid(h, w) / lower_right_corner

        # 左上端からの距離を計算。
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]

        # 右下端からの距離を計算。
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]

        # 左上と右下の距離のうち小さい方を選択し、最終的な境界距離を計算。
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]

        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        """
        画像と低解像度の特徴マップ間の重み付けを計算するメソッド。
        :param h: 元の画像の高さ。
        :param w: 元の画像の幅。
        :param Ly: 低解像度特徴マップの高さ。
        :param Lx: 低解像度特徴マップの幅。
        :param device: 使用するデバイス（CPUまたはGPU）。
        :return: 計算された重み付け。
        """
        # 画像の境界までの距離を基に重み付けを計算する。
        weighting = self.delta_border(h, w)

        # 重み付けを指定された範囲内でクリップする。
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"])

        # 重み付けを適切な形状に変形し、指定されたデバイスに移動する。
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        # タイブレーカーが有効な場合、追加の重み付けを計算する。
        if self.split_input_params["tie_braker"]:
            # 低解像度特徴マップの境界までの距離を基に重み付けを計算する。
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            # 重み付けを適切な形状に変形し、指定されたデバイスに移動する。
            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)

            # 元の重み付けと追加の重み付けを組み合わせる。
            weighting = weighting * L_weighting

        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        """
        画像の畳み込みと展開（fold/unfold）操作を行うメソッド。
        画像の畳み込み（fold）と展開（unfold）操作を提供します。これにより、画像をカーネルサイズに基づいた小さなパッチに分割し、
        それらのパッチを処理した後に元の画像サイズに戻すことができます。また、異なるアップスケール係数（uf）とダウンスケール係数（df）に
        対応するための複数の設定が用意されています。これにより、畳み込みと展開の操作を様々な解像度の画像に適用することが可能です。
        重み付けと正規化は、これらの操作の均等性を確保するために使用されます。
        :param x: 入力画像（サイズは (bs, c, h, w)）。
        :param kernel_size: カーネルのサイズ。
        :param stride: ストライドのサイズ。
        :param uf: 展開のアップスケール係数。
        :param df: 畳み込みのダウンスケール係数。
        :return: 畳み込みと展開の操作、正規化、重み付け。
        """
        bs, nc, h, w = x.shape

        # 画像内のクロップ数を計算
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        # 展開と畳み込みのパラメータ設定
        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            # 重み付けと正規化
            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        # ufが1より大きく、dfが1の場合
        elif uf > 1 and df == 1:
            # 展開と畳み込みの設定
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            # 重み付けと正規化
            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        # dfが1より大きく、ufが1の場合
        elif df > 1 and uf == 1:
            # 展開と畳み込みの設定
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            # 重み付けと正規化
            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False):
        """
        バッチデータから入力を取得し、必要に応じて処理するメソッド。
        :param batch: データバッチ。
        :param k: バッチ内のキー。
        :param return_first_stage_outputs: 最初のステージの出力を返すかどうか。
        :param force_c_encode: 条件付け情報を強制的にエンコードするかどうか。
        :param cond_key: 条件付けに使用するキー。
        :param return_original_cond: 元の条件付け情報を返すかどうか。
        :param bs: 処理するバッチサイズ。
        :param return_x: 入力画像を返すかどうか。
        :return: 処理された入力データのリスト。
        """
        # スーパークラスのget_inputを使用して入力を取得。
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        # 最初のステージのエンコーダーを使用してポステリアを計算。
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        # 条件付けキーが設定されており、強制的にnull条件付けを行わない場合の処理。
        if self.model.conditioning_key is not None and not self.force_null_conditioning:
            if cond_key is None:
                cond_key = self.cond_stage_key

            # 条件付けデータの取得。
            if cond_key != self.first_stage_key:
                # 異なる種類の条件付けデータを取得。
                if cond_key in ['caption', 'coordinates_bbox', "txt"]:
                    xc = batch[cond_key]
                elif cond_key in ['class_label', 'cls']:
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x

            # 条件付けデータのエンコード処理。
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            # 位置エンコーディングの使用。
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            # null条件付けの場合の処理。
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}

        # 出力リストの作成。
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        """
        エンコードされた潜在変数を最初のステージのデコーダーを使用して元の画像にデコードするメソッド。
        :param z: エンコードされた潜在変数。
        :param predict_cids: コードブックインデックスを予測するかどうか。
        :param force_not_quantize: 量子化を行わないかどうかのフラグ。
        :return: デコードされた画像。
        """
        # コードブックインデックスを予測する場合の処理。
        if predict_cids:
            # zが4次元の場合（バッチサイズ、高さ、幅、チャネル）、最大確率のインデックスを取得。
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            # インデックスに基づいてコードブックエントリを取得し、適切な形状に再配置。
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        # スケールファクターを用いて潜在変数zをスケーリング。
        z = 1. / self.scale_factor * z

        # 最初のステージのデコーダーでデコードを行い、結果を返す。
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        """
        最初のステージのモデルを使用して入力xをエンコードするメソッド。
        :param x: 入力画像。
        :return: エンコードされた潜在変数。
        """
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        """
        トレーニングと検証の共有ステップを定義するメソッド。
        入力データと条件付けデータを取得し、損失を計算する。
        :param batch: 入力バッチデータ。
        :return: 計算された損失。
        """
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        """
        モデルのフォワードパスを定義するメソッド。
        ランダムな拡散タイムステップを選択し、条件付けデータを使用して損失を計算する。
        :param x: 入力画像。
        :param c: 条件付けデータ。
        :return: 計算された損失。
        """
        # ランダムな拡散タイムステップを選択
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        # 条件付けキーが設定されている場合の処理
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                # 学習済みの条件付け情報を取得
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: このオプションを廃止するか検討
                # 条件付けスケジュールに従って拡散
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        モデルにノイズ付き入力と条件付けデータを適用し、再構築された画像を取得するメソッド。
        :param x_noisy: ノイズが加えられた入力画像。
        :param t: 拡散プロセスのタイムステップ。
        :param cond: 条件付けデータ。
        :param return_ids: 返却するのがIDかどうか。
        :return: 再構築された画像、またはモデルの出力。
        """
        # 条件付けデータが辞書型の場合（ハイブリッドケース）
        if isinstance(cond, dict):
            pass  # 何もせずにそのまま使用
        else:
            # 条件付けデータがリストでない場合、リストに変換
            if not isinstance(cond, list):
                cond = [cond]
            # モデルの条件付けキーに応じて、条件付けデータを辞書に変換
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # モデルにノイズ付き入力と条件付けデータを適用
        x_recon = self.model(x_noisy, t, **cond)

        # モデルの出力がタプルの場合、return_idsがFalseなら最初の要素を返す
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        ノイズ付きの入力からノイズを予測するメソッド。
        :param x_t: 拡散プロセス中の特定のステップにおけるノイズが加えられた画像。
        :param t: 拡散プロセス中のタイムステップ。
        :param pred_xstart: 再構築されたノイズがない元の画像。
        :return: 予測されたノイズ。
        """
        # 予測されたノイズを計算するための式。アルファ値を用いてスケーリングされる。
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        変分下限の事前KL項をビット毎次元で取得するメソッド。
        この項はエンコーダーにのみ依存し、最適化することはできない。
        :param x_start: ノイズがない入力画像。
        :return: バッチ要素ごとのKL値のバッチ（ビット単位）。
        """
        batch_size = x_start.shape[0]
        # 最後のタイムステップを選択
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        # qの平均と分散を取得
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        # 事前KL項を計算
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        # 平均値を計算し、自然対数の底で割ってビット毎次元に変換
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        """
        モデルの損失を計算するメソッド。
        :param x_start: 元のノイズがない画像。
        :param cond: 条件付けデータ。
        :param t: 拡散のタイムステップ。
        :param noise: 加えるノイズ（指定されていなければランダムに生成）。
        :return: 計算された損失と、損失に関する詳細情報を含む辞書。
        """
        # ノイズを生成または指定されたノイズを使用
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 拡散プロセスをシミュレートしてノイズ付き画像を生成
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # モデルにノイズ付き画像と条件付けデータを適用
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # 損失のターゲットを決定
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # シンプルな損失を計算
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # ログ分散に基づく追加の損失計算
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # 変分下限の損失を計算
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        """
        モデルの予測平均と分散を計算するメソッド。
        :param x: ノイズ付きの入力画像。
        :param c: 条件付けデータ。
        :param t: 拡散のタイムステップ。
        :param clip_denoised: 再構築された画像をクリップするかどうか。
        :param return_codebook_ids: コードブックのIDを返すかどうか。
        :param quantize_denoised: 再構築された画像を量子化するかどうか。
        :param return_x0: 元の画像を返すかどうか。
        :param score_corrector: スコア補正関数（オプション）。
        :param corrector_kwargs: スコア補正関数のキーワード引数。
        :return: 予測平均、分散、（オプションで）ログ分散、その他のオプション出力。
        """
        t_in = t
        # モデルにノイズ付き入力と条件付けデータを適用
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        # スコア補正関数が指定されている場合、適用
        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        # コードブックIDの処理
        if return_codebook_ids:
            model_out, logits = model_out

        # パラメータ化に基づいて再構築された画像を計算
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        # 再構築された画像のクリップ処理
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        # 再構築された画像の量子化処理
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        # 予測平均、分散、ログ分散を計算
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        # オプションの出力を返す
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        """
        特定のタイムステップにおいてモデルからサンプルを生成するメソッド。
        :param x: ノイズ付き入力画像。
        :param c: 条件付けデータ。
        :param t: 拡散プロセスのタイムステップ。
        :param clip_denoised: 再構築画像をクリップするかどうか。
        :param repeat_noise: ノイズを繰り返すかどうか。
        :param return_codebook_ids: コードブックIDを返すかどうか。
        :param quantize_denoised: 再構築画像を量子化するかどうか。
        :param return_x0: 元の画像を返すかどうか。
        :param temperature: ノイズの温度。
        :param noise_dropout: ノイズのドロップアウト率。
        :param score_corrector: スコア補正関数。
        :param corrector_kwargs: スコア補正関数のキーワード引数。
        :return: 生成されたサンプル、およびオプションの追加出力。
        """
        b, *_, device = *x.shape, x.device
        # 予測平均と分散を取得
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

        # 出力の取得と処理
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        # ノイズの生成と適用
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # tが0の時はノイズを加えない
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # サンプルの生成
        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            if self.cond_stage_key in ["class_label", "cls"]:
                xc = self.cond_stage_model.get_unconditional_conditioning(batch_size, device=self.device)
                return self.get_learned_conditioning(xc)
            else:
                raise NotImplementedError("todo")
        if isinstance(c, list):  # in case the encoder gives us a list
            for i in range(len(c)):
                c[i] = repeat(c[i], '1 ... -> b ...', b=batch_size).to(self.device)
        else:
            c = repeat(c, '1 ... -> b ...', b=batch_size).to(self.device)
        return c

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', "cls"]:
                try:
                    xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                    log['conditioning'] = xc
                except KeyError:
                    # probably no "human_label" in batch
                    pass
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            if self.model.conditioning_key == "crossattn-adm":
                uc = {"c_crossattn": [uc], "c_adm": c["c_adm"]}
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        if inpaint:
            # make a simple center square
            b, h, w = z.shape[0], z.shape[2], z.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):
                samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_inpainting"] = x_samples
            log["mask"] = mask

            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if not self.sequential_cross_attn:
                cc = torch.cat(c_crossattn, 1)
            else:
                cc = c_crossattn
            if hasattr(self, "scripted_diffusion_model"):
                # TorchScript changes names of the arguments
                # with argument cc defined as context=cc scripted model will produce
                # an error: RuntimeError: forward() is missing value for argument 'argument_3'.
                out = self.scripted_diffusion_model(x, t, cc)
            else:
                out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class LatentUpscaleDiffusion(LatentDiffusion):
    def __init__(self, *args, low_scale_config, low_scale_key="LR", noise_level_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        # assumes that neither the cond_stage nor the low_scale_model contain trainable params
        assert not self.cond_stage_trainable
        self.instantiate_low_stage(low_scale_config)
        self.low_scale_key = low_scale_key
        self.noise_level_key = noise_level_key

    def instantiate_low_stage(self, config):
        model = instantiate_from_config(config)
        self.low_scale_model = model.eval()
        self.low_scale_model.train = disabled_train
        for param in self.low_scale_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, log_mode=False):
        if not log_mode:
            z, c = super().get_input(batch, k, force_c_encode=True, bs=bs)
        else:
            z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                                  force_c_encode=True, return_original_cond=True, bs=bs)
        x_low = batch[self.low_scale_key][:bs]
        x_low = rearrange(x_low, 'b h w c -> b c h w')
        x_low = x_low.to(memory_format=torch.contiguous_format).float()
        zx, noise_level = self.low_scale_model(x_low)
        if self.noise_level_key is not None:
            # get noise level from batch instead, e.g. when extracting a custom noise level for bsr
            raise NotImplementedError('TODO')

        all_conds = {"c_concat": [zx], "c_crossattn": [c], "c_adm": noise_level}
        if log_mode:
            # TODO: maybe disable if too expensive
            x_low_rec = self.low_scale_model.decode(zx)
            return z, all_conds, x, xrec, xc, x_low, x_low_rec, noise_level
        return z, all_conds

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True,
                   unconditional_guidance_scale=1., unconditional_guidance_label=None, use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc, x_low, x_low_rec, noise_level = self.get_input(batch, self.first_stage_key, bs=N,
                                                                          log_mode=True)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["x_lr"] = x_low
        log[f"x_lr_rec_@noise_levels{'-'.join(map(lambda x: str(x), list(noise_level.cpu().numpy())))}"] = x_low_rec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', 'cls']:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_tmp = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            # TODO explore better "unconditional" choices for the other keys
            # maybe guide away from empty text label and highest noise level and maximally degraded zx?
            uc = dict()
            for k in c:
                if k == "c_crossattn":
                    assert isinstance(c[k], list) and len(c[k]) == 1
                    uc[k] = [uc_tmp]
                elif k == "c_adm":  # todo: only run with text-based guidance?
                    assert isinstance(c[k], torch.Tensor)
                    #uc[k] = torch.ones_like(c[k]) * self.low_scale_model.max_noise_level
                    uc[k] = c[k]
                elif isinstance(c[k], list):
                    uc[k] = [c[k][i] for i in range(len(c[k]))]
                else:
                    uc[k] = c[k]

            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        return log


class LatentFinetuneDiffusion(LatentDiffusion):
    """
         Basis for different finetunas, such as inpainting or depth2image
         To disable finetuning mode, set finetune_keys to None
    """

    def __init__(self,
                 concat_keys: tuple,
                 finetune_keys=("model.diffusion_model.input_blocks.0.0.weight",
                                "model_ema.diffusion_modelinput_blocks00weight"
                                ),
                 keep_finetune_dims=4,
                 # if model was trained without concat mode before and we would like to keep these channels
                 c_concat_log_start=None,  # to log reconstruction of c_concat codes
                 c_concat_log_end=None,
                 *args, **kwargs
                 ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", list())
        super().__init__(*args, **kwargs)
        self.finetune_keys = finetune_keys
        self.concat_keys = concat_keys
        self.keep_dims = keep_finetune_dims
        self.c_concat_log_start = c_concat_log_start
        self.c_concat_log_end = c_concat_log_end
        if exists(self.finetune_keys): assert exists(ckpt_path), 'can only finetune from a given checkpoint'
        if exists(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

            # make it explicit, finetune by including extra input channels
            if exists(self.finetune_keys) and k in self.finetune_keys:
                new_entry = None
                for name, param in self.named_parameters():
                    if name in self.finetune_keys:
                        print(
                            f"modifying key '{name}' and keeping its original {self.keep_dims} (channels) dimensions only")
                        new_entry = torch.zeros_like(param)  # zero init
                assert exists(new_entry), 'did not find matching parameter to modify'
                new_entry[:, :self.keep_dims, ...] = sd[k]
                sd[k] = new_entry

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', 'cls']:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if not (self.c_concat_log_start is None and self.c_concat_log_end is None):
            log["c_concat_decoded"] = self.decode_first_stage(c_cat[:, self.c_concat_log_start:self.c_concat_log_end])

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                         batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                 batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc_full,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log


class LatentInpaintDiffusion(LatentFinetuneDiffusion):
    """
    can either run as pure inpainting model (only concat mode) or with mixed conditionings,
    e.g. mask as concat and text via cross-attn.
    To disable finetuning mode, set finetune_keys to None
     """

    def __init__(self,
                 concat_keys=("mask", "masked_image"),
                 masked_image_key="masked_image",
                 *args, **kwargs
                 ):
        super().__init__(concat_keys, *args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for inpainting'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        c_cat = list()
        for ck in self.concat_keys:
            cc = rearrange(batch[ck], 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            bchw = z.shape
            if ck != self.masked_image_key:
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super(LatentInpaintDiffusion, self).log_images(*args, **kwargs)
        log["masked_image"] = rearrange(args[0]["masked_image"],
                                        'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()
        return log


class LatentDepth2ImageDiffusion(LatentFinetuneDiffusion):
    """
    condition on monocular depth estimation
    """

    def __init__(self, depth_stage_config, concat_keys=("midas_in",), *args, **kwargs):
        super().__init__(concat_keys=concat_keys, *args, **kwargs)
        self.depth_model = instantiate_from_config(depth_stage_config)
        self.depth_stage_key = concat_keys[0]

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for depth2img'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        assert len(self.concat_keys) == 1
        c_cat = list()
        for ck in self.concat_keys:
            cc = batch[ck]
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            cc = self.depth_model(cc)
            cc = torch.nn.functional.interpolate(
                cc,
                size=z.shape[2:],
                mode="bicubic",
                align_corners=False,
            )

            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            cc = 2. * (cc - depth_min) / (depth_max - depth_min + 0.001) - 1.
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super().log_images(*args, **kwargs)
        depth = self.depth_model(args[0][self.depth_stage_key])
        depth_min, depth_max = torch.amin(depth, dim=[1, 2, 3], keepdim=True), \
                               torch.amax(depth, dim=[1, 2, 3], keepdim=True)
        log["depth"] = 2. * (depth - depth_min) / (depth_max - depth_min) - 1.
        return log


class LatentUpscaleFinetuneDiffusion(LatentFinetuneDiffusion):
    """
        condition on low-res image (and optionally on some spatial noise augmentation)
    """
    def __init__(self, concat_keys=("lr",), reshuffle_patch_size=None,
                 low_scale_config=None, low_scale_key=None, *args, **kwargs):
        super().__init__(concat_keys=concat_keys, *args, **kwargs)
        self.reshuffle_patch_size = reshuffle_patch_size
        self.low_scale_model = None
        if low_scale_config is not None:
            print("Initializing a low-scale model")
            assert exists(low_scale_key)
            self.instantiate_low_stage(low_scale_config)
            self.low_scale_key = low_scale_key

    def instantiate_low_stage(self, config):
        model = instantiate_from_config(config)
        self.low_scale_model = model.eval()
        self.low_scale_model.train = disabled_train
        for param in self.low_scale_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for upscaling-ft'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        assert len(self.concat_keys) == 1
        # optionally make spatial noise_level here
        c_cat = list()
        noise_level = None
        for ck in self.concat_keys:
            cc = batch[ck]
            cc = rearrange(cc, 'b h w c -> b c h w')
            if exists(self.reshuffle_patch_size):
                assert isinstance(self.reshuffle_patch_size, int)
                cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                               p1=self.reshuffle_patch_size, p2=self.reshuffle_patch_size)
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            if exists(self.low_scale_model) and ck == self.low_scale_key:
                cc, noise_level = self.low_scale_model(cc)
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        if exists(noise_level):
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c], "c_adm": noise_level}
        else:
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super().log_images(*args, **kwargs)
        log["lr"] = rearrange(args[0]["lr"], 'b h w c -> b c h w')
        return log


class ImageEmbeddingConditionedLatentDiffusion(LatentDiffusion):
    def __init__(self, embedder_config, embedding_key="jpg", embedding_dropout=0.5,
                 freeze_embedder=True, noise_aug_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_key = embedding_key
        self.embedding_dropout = embedding_dropout
        self._init_embedder(embedder_config, freeze_embedder)
        self._init_noise_aug(noise_aug_config)

    def _init_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def _init_noise_aug(self, config):
        if config is not None:
            # use the KARLO schedule for noise augmentation on CLIP image embeddings
            noise_augmentor = instantiate_from_config(config)
            assert isinstance(noise_augmentor, nn.Module)
            noise_augmentor = noise_augmentor.eval()
            noise_augmentor.train = disabled_train
            self.noise_augmentor = noise_augmentor
        else:
            self.noise_augmentor = None

    def get_input(self, batch, k, cond_key=None, bs=None, **kwargs):
        outputs = LatentDiffusion.get_input(self, batch, k, bs=bs, **kwargs)
        z, c = outputs[0], outputs[1]
        img = batch[self.embed_key][:bs]
        img = rearrange(img, 'b h w c -> b c h w')
        c_adm = self.embedder(img)
        if self.noise_augmentor is not None:
            c_adm, noise_level_emb = self.noise_augmentor(c_adm)
            # assume this gives embeddings of noise levels
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        if self.training:
            c_adm = torch.bernoulli((1. - self.embedding_dropout) * torch.ones(c_adm.shape[0],
                                                                               device=c_adm.device)[:, None]) * c_adm
        all_conds = {"c_crossattn": [c], "c_adm": c_adm}
        noutputs = [z, all_conds]
        noutputs.extend(outputs[2:])
        return noutputs

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, **kwargs):
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True,
                                           return_original_cond=True)
        log["inputs"] = x
        log["reconstruction"] = xrec
        assert self.model.conditioning_key is not None
        assert self.cond_stage_key in ["caption", "txt"]
        xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
        log["conditioning"] = xc
        uc = self.get_unconditional_conditioning(N, kwargs.get('unconditional_guidance_label', ''))
        unconditional_guidance_scale = kwargs.get('unconditional_guidance_scale', 5.)

        uc_ = {"c_crossattn": [uc], "c_adm": c["c_adm"]}
        ema_scope = self.ema_scope if kwargs.get('use_ema_scope', True) else nullcontext
        with ema_scope(f"Sampling"):
            samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=True,
                                             ddim_steps=kwargs.get('ddim_steps', 50), eta=kwargs.get('ddim_eta', 0.),
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_, )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samplescfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log
