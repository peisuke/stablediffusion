import importlib

import torch
from torch import optim
import numpy as np

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


def autocast(f):
    # デコレーター：関数を自動的にミックス精度で実行するために使用されます。
    def do_autocast(*args, **kwargs):
        # PyTorchの自動キャストを有効にして、指定された関数を実行します。
        with torch.cuda.amp.autocast(enabled=True,
                                     dtype=torch.get_autocast_gpu_dtype(),
                                     cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)

    return do_autocast


def log_txt_as_img(wh, xc, size=10):
    # この関数はテキストをイメージに変換します。
    # wh は (幅, 高さ) のタプル、xc はプロットするキャプションのリストです。
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    # xが4次元のテンソルで、2次元目のサイズが3より大きい場合にTrueを返します。
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    # xが画像のテンソル表現かどうかを判定します。
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    # xがNoneではない場合にTrueを返します。
    return x is not None


def default(val, d):
    # valが存在する場合はvalを返し、そうでない場合はdを返します。
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    テンソルの非バッチ次元にわたる平均を取ります。
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    # モデルのパラメータの総数を計算します。
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    # 設定からオブジェクトをインスタンス化します。
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    # 文字列からオブジェクトを取得します
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    # AdamW最適化アルゴリズムをベースに、EMA（Exponential Moving Average）を組み込んだカスタム最適化器
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """
        AdamWのパラメータとEMA関連の設定を初期化します。
        :param params: 最適化対象のパラメータ
        :param lr: 学習率
        :param betas: Adamのベータ係数（モーメンタム項）
        :param eps: 数値安定性のための小さな値
        :param weight_decay: 重み減衰率
        :param amsgrad: AMSGradバリアントを使用するかどうか
        :param ema_decay: EMAの減衰率
        :param ema_power: EMAのパワー（調整用）
        :param param_names: パラメータ名（オプション）
        """

        # 入力値のバリデーション
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                        ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        # オプティマイザの状態を設定します。
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        単一の最適化ステップを実行します。
        :param closure: モデルを再評価して損失を返すクロージャ
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 各種パラメータと状態を初期化
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            # EMAの計算と更新
            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss