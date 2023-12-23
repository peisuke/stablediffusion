import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False
                 ):
        """
        オートエンコーダの初期化を行います。
        :param ddconfig: エンコーダーとデコーダーの設定
        :param lossconfig: 損失関数の設定
        :param embed_dim: 埋め込みの次元数
        :param ckpt_path: チェックポイントのパス（省略可能）
        :param ignore_keys: チェックポイントから無視するキー（省略可能）
        :param image_key: バッチ内の画像のキー
        :param colorize_nlabels: 色付けのためのラベル数（省略可能）
        :param monitor: モニタリング用のオブジェクト（省略可能）
        :param ema_decay: EMAの減衰率（省略可能）
        :param learn_logvar: 対数分散を学習するかどうか（省略可能）
        """
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        """
        チェックポイントからモデルを初期化します。
        :param path: チェックポイントのパス
        :param ignore_keys: 無視するキーのリスト
        """
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        """
        EMAのスコープを管理するコンテキストマネージャーです。
        このスコープ内では、モデルのパラメータがEMAの値に一時的に切り替わります。
        :param context: コンテキスト名（ログ出力用、省略可能）
        """

        if self.use_ema:
            # EMAを使用する場合、現在のモデルのパラメータを保存し、EMAの値に置き換えます。
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                # コンテキストが提供されている場合、ログにEMA重みへの切り替えを記録します。
                print(f"{context}: Switched to EMA weights")

        try:
            # コンテキストマネージャーの利用側での処理を実行します。
            # ここで 'yield' される値は呼び出し元で使用されますが、この例では 'None' を返しています。
            yield None
        finally:
            # コンテキストマネージャーを抜ける際に必ず実行される処理。
            if self.use_ema:
                # EMAを使用している場合、EMAの値を元に戻して、元のトレーニングパラメータを復元します。
                self.model_ema.restore(self.parameters())
                if context is not None:
                    # コンテキストが提供されている場合、ログにトレーニング重みへの復元を記録します。
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        """
        入力データをエンコードする関数。
        :param x: 入力データ
        :return: エンコードされたデータの事後分布
        """
        # 入力データxをエンコーダーを通して処理します。
        h = self.encoder(x)
        # エンコードされた特徴量hに量子化層を適用します。
        moments = self.quant_conv(h)
        # 得られたモーメントから対角ガウス分布（Diagonal Gaussian Distribution）の事後分布を生成します。
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        """
        エンコードされたデータをデコードする関数。
        :param z: エンコードされたデータ
        :return: デコードされたデータ
        """
        # エンコードされたデータzに対して、逆量子化層を適用します。
        z = self.post_quant_conv(z)
        # 逆量子化されたデータをデコーダーを通して処理し、デコードされたデータを得ます。
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        モデルのフォワードパス。
        :param input: 入力データ
        :param sample_posterior: Trueの場合、事後分布からサンプリングを行う
        :return: デコードされたデータと事後分布
        """
        # 入力データをエンコードして事後分布を得ます。
        posterior = self.encode(input)
        if sample_posterior:
            # sample_posteriorがTrueの場合、事後分布からサンプリングを行い、エンコードされたデータを得ます。
            z = posterior.sample()
        else:
            # Falseの場合、事後分布のモード（最も確率の高い値）をエンコードされたデータとして使用します。
            z = posterior.mode()
        # エンコードされたデータをデコードします。
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        """
        バッチから特定のキーに対応する入力データを取得し、必要な形式に変換する関数。
        :param batch: データバッチ（辞書形式）
        :param k: 取得するデータのキー
        :return: 変換後の入力データ
        """
        # バッチからキーkに対応するデータを取得します。
        x = batch[k]
        # データが3次元の場合（通常、高さ×幅×チャンネル）、新たな次元を追加して4次元にします。
        # これはバッチ次元が欠けている場合にバッチ次元を追加するための処理です。
        if len(x.shape) == 3:
            x = x[..., None]
        # データの次元を並べ替えます（バッチ、チャンネル、高さ、幅の順）。
        # これはPyTorchで一般的なデータ形式（NCHW形式）に合わせるためです。
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        # 変換されたデータを返します。
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        トレーニングステップの定義。
        :param batch: 現在のバッチ
        :param batch_idx: 現在のバッチのインデックス
        :param optimizer_idx: 使用するオプティマイザのインデックス
        :return: 損失値
        """
        # バッチから入力データを取得します。
        inputs = self.get_input(batch, self.image_key)
        # 入力データをモデルに通して、再構成されたデータと事後分布を得ます。
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # オプティマイザのインデックスが0の場合、エンコーダ、デコーダ、およびlogvarをトレーニングします。
            # 損失計算関数を呼び出し、損失値とログ記録用の辞書を取得します。
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            # 損失値とログ情報を記録します。
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # オプティマイザのインデックスが1の場合、識別器をトレーニングします。
            # 損失計算関数を呼び出し、損失値とログ記録用の辞書を取得します。
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            # 損失値とログ情報を記録します。
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        """
        バリデーションステップの定義。
        :param batch: 現在のバッチ
        :param batch_idx: 現在のバッチのインデックス
        :return: ログディクショナリ
        """
        # 通常のバリデーションステップを実行し、ロギング用のディクショナリを取得します。
        log_dict = self._validation_step(batch, batch_idx)
        # EMA（指数移動平均）のスコープ内でバリデーションステップを再実行します。
        with self.ema_scope():
            # EMAを使用したバリデーションの結果を取得します。
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        # 通常のバリデーションの結果を返します。
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        """
        内部バリデーションステップのヘルパー関数。
        :param batch: 現在のバッチ
        :param batch_idx: 現在のバッチのインデックス
        :param postfix: ログキーに付加するポストフィックス（EMA使用時に区別するため）
        :return: ログディクショナリ
        """
        # バッチから入力データを取得します。
        inputs = self.get_input(batch, self.image_key)
        # 入力データをモデルに通して、再構成されたデータと事後分布を得ます。
        reconstructions, posterior = self(inputs)
        # エンコーダ、デコーダの損失を計算します。
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+postfix)
        # 識別器の損失を計算します。
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val"+postfix)
        # 損失値とログ情報を記録します。
        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        """
        モデルのオプティマイザを設定する関数。
        :return: オプティマイザのリスト
        """
        # 学習率を取得します。
        lr = self.learning_rate
        # エンコーダ、デコーダ、量子化層のパラメータをオプティマイザのパラメータリストに追加します。
        ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                         list(self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        # learn_logvarがTrueの場合、損失関数のlogvarパラメータも学習対象に追加します。
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        # エンコーダとデコーダのオプティマイザをAdamで設定します。
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        # 識別器のオプティマイザもAdamで設定します。
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        # 2つのオプティマイザをリストとして返します。
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """
        モデルの最後の層を取得する関数。
        :return: デコーダの最後の畳み込み層の重み
        """
        # デコーダの最後の畳み込み層の重みを返します。
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        """
        画像のロギングを行う関数。勾配計算は行わない。
        :param batch: 現在のバッチ
        :param only_inputs: Trueの場合、入力データのみをログに記録
        :param log_ema: Trueの場合、EMAを使用した結果もログに記録
        :param kwargs: その他のパラメータ
        :return: ログデータ（辞書形式）
        """
        # ログデータを格納する辞書を初期化
        log = dict()
        # バッチから入力データを取得し、モデルが使用するデバイスに転送
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        if not only_inputs:
            # only_inputsがFalseの場合、入力データをモデルに通して再構成し、事後分布を得る
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # チャンネル数が3より多い場合、ランダムな投影を使用して色付け
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # ランダムなノイズからサンプルを生成し、デコードしてログに記録
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            # 再構成された画像をログに記録
            log["reconstructions"] = xrec

            if log_ema or self.use_ema:
                # log_emaまたはuse_emaがTrueの場合、EMA重みを使用して再度ログ記録
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # チャンネル数が3より多い場合、色付け処理
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    # EMAを使用して生成されたサンプルと再構成画像をログに記録
                    log["samples_ema"] = self.decode(torch.randn_like(posterior_ema.sample()))
                    log["reconstructions_ema"] = xrec_ema
        # 入力データをログに記録
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        """
        セグメンテーションマップをRGB画像に変換する関数。
        :param x: 入力データ（セグメンテーションマップ）
        :return: RGB画像
        """
        # 入力がセグメンテーションデータであることを確認
        assert self.image_key == "segmentation"
        # カラー化用のバッファがまだ存在しない場合は作成
        if not hasattr(self, "colorize"):
            # カラー化用の重みをランダムに初期化し、バッファに登録
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        # 3チャンネルのRGB画像に変換するために畳み込みを適用
        x = F.conv2d(x, weight=self.colorize)
        # 結果を[-1, 1]の範囲に正規化
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

