import torch

from ldm.modules.midas.api import load_midas_transform


class AddMiDaS(object):
    def __init__(self, model_type):
        # クラスの初期化メソッド
        super().__init__()
        # MiDaSモデル用の変換をロードします。model_typeに基づいて異なる変換が適用される可能性があります。
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        # PyTorchテンソルをNumPy配列に変換するメソッド
        # xは[-1, 1]の範囲で、これを[0, 1]の範囲に変換し、CPU上のNumPy配列にします。
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        # NumPy配列をPyTorchテンソルに変換するメソッド
        # xは[0, 1]の範囲で、これを[-1, 1]の範囲に変換します。
        x = torch.from_numpy(x) * 2 - 1.
        return x

    def __call__(self, sample):
        # オブジェクトが関数のように呼び出された時に実行されるメソッド
        # sample辞書の'jpg'キーに対応するテンソルを取得し、MiDaSモデル用に変換します。
        # sample['jpg'] はこの時点で [-1, 1] の範囲のテンソル（hwc形式）
        x = self.pt2np(sample['jpg'])
        # MiDaS変換を適用します。
        x = self.transform({"image": x})["image"]
        # 変換したデータをsample辞書に 'midas_in' キーとして追加します。
        sample['midas_in'] = x
        return sample