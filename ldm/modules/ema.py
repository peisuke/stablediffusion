import torch
from torch import nn

class LitEma(nn.Module):
    """
    PyTorchモジュールを拡張したEMA（指数移動平均）モジュール。
    モデルのパラメータに対してEMAを適用し、平滑化したパラメータを保持する。
    """

    def __init__(self, model, decay=0.9999, use_num_upates=True):
        """
        EMAモジュールの初期化。
        :param model: EMAを適用する対象のモデル
        :param decay: EMAの減衰率。0から1の間の値。
        :param use_num_upates: 更新回数を考慮するかどうかのフラグ
        """
        super().__init__()
        # 減衰率が0と1の間でなければエラーを発生させる
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        # モデルのパラメータ名とEMA用のバッファ名をマッピングする辞書
        self.m_name2s_name = {}
        # 減衰率をバッファとして登録
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        # 更新回数を管理するバッファを登録。use_num_updatesがTrueの場合は0から開始、Falseの場合は-1
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
                             else torch.tensor(-1, dtype=torch.int))

        # モデルの各パラメータに対して
        for name, p in model.named_parameters():
            # 勾配が必要なパラメータのみを処理
            if p.requires_grad:
                # パラメータ名から'.'を削除（バッファ名に'.'は使用できないため）
                s_name = name.replace('.', '')
                # パラメータ名とバッファ名のマッピングを更新
                self.m_name2s_name.update({name: s_name})
                # パラメータのクローンをバッファとして登録
                self.register_buffer(s_name, p.clone().detach().data)

        # EMA適用後のパラメータを保持するリスト
        self.collected_params = []

    def reset_num_updates(self):
        """
        更新回数カウンタをリセットする関数。
        """
        # 現在の更新回数を削除
        del self.num_updates
        # 更新回数を0で初期化し、バッファとして再登録
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))
    
    def forward(self, model):
        """
        EMAの更新を行うフォワード関数。
        :param model: EMAの適用対象モデル
        """
        # 減衰率を取得
        decay = self.decay
    
        # 更新回数が0以上の場合、更新回数をインクリメント
        if self.num_updates >= 0:
            self.num_updates += 1
            # 減衰率を更新回数に応じて調整
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
    
        # 1から減衰率を引いた値を計算
        one_minus_decay = 1.0 - decay
    
        # 勾配計算を行わないコンテキスト内で処理
        with torch.no_grad():
            # モデルのパラメータを取得
            m_param = dict(model.named_parameters())
            # EMA用のシャドウパラメータを取得
            shadow_params = dict(self.named_buffers())
    
            # 各パラメータについて
            for key in m_param:
                # 勾配が必要なパラメータのみを処理
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    # シャドウパラメータの型をモデルのパラメータの型に合わせる
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    # シャドウパラメータを更新
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    # 勾配が不要なパラメータは処理しないことを確認
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        """
        EMAモジュールのシャドウパラメータをモデルのパラメータにコピーする関数。
        :param model: 更新対象のモデル
        """
        # モデルのパラメータを取得
        m_param = dict(model.named_parameters())
        # EMAモジュールのシャドウパラメータを取得
        shadow_params = dict(self.named_buffers())
        # 各パラメータについて
        for key in m_param:
            # 勾配が必要なパラメータのみを処理
            if m_param[key].requires_grad:
                # シャドウパラメータの値をモデルのパラメータにコピー
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                # 勾配が不要なパラメータは処理しないことを確認
                assert not key in self.m_name2s_name
    
    def store(self, parameters):
        """
        現在のパラメータを保存するための関数。
        :param parameters: 保存するパラメータのイテラブル
        """
        # パラメータのクローンを作成し、保存
        self.collected_params = [param.clone() for param in parameters]
    
    def restore(self, parameters):
        """
        `store`メソッドで保存したパラメータを元に戻す関数。
        EMAパラメータを使用してモデルを評価する際に、元の最適化プロセスに影響を与えずにパラメータを復元するのに役立つ。
        `copy_to`メソッドの前にパラメータを保存し、評価（またはモデルの保存）後にこれを使用して元のパラメータを復元する。
        :param parameters: 更新するパラメータのイテラブル
        """
        # 保存したパラメータと現在のパラメータを順に処理
        for c_param, param in zip(self.collected_params, parameters):
            # 保存されたパラメータの値を現在のパラメータにコピー
            param.data.copy_(c_param.data)
    