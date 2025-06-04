import numpy as np
import torch as t
from typing import Dict, List, Any, Tuple, Union

# --- 異常タイプごとのデフォルトパラメータ定義 ---
# 全ての異常タイプに共通するデフォルトパラメータ
COMMON_ANOMALY_PARAMS: Dict[str, Any] = {
    'min_range': 20,
    'min_features': 1,
    'max_features': 5,
}

# 各異常タイプに固有のデフォルトパラメータ。
# 共通パラメータはここで上書きまたは追加されます。
ANOMALY_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'spike': {'scale': 1},
    'flip': {'anomaly_range': 200, 'fast_sampling': False},
    'speedup': {'frequency': [0.5, 2], 'fast_sampling': False},
    'noise': {'scale': 0.1},
    'cutoff': {},
    'average': {'ma_window': 20, 'anomaly_range': 200, 'fast_sampling': False},
    'scale': {'scale': 1},
    'wander': {'scale': 1},
    'contextual': {'scale': 1},
    'upsidedown': {},
    'mixture': {}
}

class AnomalyInjector:
    """
    時系列データに様々な種類の異常を注入するための基底クラス。
    各異常注入メソッド（例: _inject_spike）は、このクラスまたは
    そのサブクラスで実装されます。
    """
    def __init__(self) -> None:
        # このクラスは直接インスタンス化されることを意図していません。
        # サブクラスが特定のパラメータを継承するために存在します。
        pass

    def _inject_spike(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウにスパイク異常を注入します。

        Parameters
        ----------
        Y_window : np.ndarray
            形状が [D, T] のウィンドウデータ。D:特徴量, T:時間。
        params : Dict[str, Any]
            スパイク異常のパラメータ (例: 'min_features', 'max_features', 'scale')。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            異常注入後のデータ、元のデータ、異常マスク。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window) # 元のデータ
        mask_temp = np.ones_like(Y_window, dtype=int) # 異常箇所は0

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_time = np.random.randint(low=0, high=window_size, size=n_anom_features)
        loc_features = np.random.randint(low=0, high=n_features, size=n_anom_features)

        # スパイクの追加
        Y_temp[loc_features, loc_time] += np.random.normal(loc=0, scale=params['scale'], size=n_anom_features)

        # マスクの更新
        mask_temp[loc_features, loc_time] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_flip(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウに反転異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                if np.random.rand() > 0.5: # 前方から選択
                    anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                    anomaly_end = np.random.randint(low=anomaly_start + params['min_range'],
                                                    high=min(anomaly_start + params['anomaly_range'], window_size) + 1)
                else: # 後方から選択
                    anomaly_end = np.random.randint(low=params['min_range'], high=window_size + 1)
                    anomaly_start = np.random.randint(low=max(0, anomaly_end - params['anomaly_range']),
                                                      high=anomaly_end - params['min_range'] + 1)

            # 範囲が不正な場合はスキップ (起こるべきではないが念のため)
            if anomaly_end <= anomaly_start:
                continue

            # シーケンスの反転
            Y_temp[loc_feature, anomaly_start:anomaly_end] = Y_temp[loc_feature, anomaly_start:anomaly_end][::-1]
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

        return Y_temp, Z_temp, mask_temp

    def _inject_speedup(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウにスピードアップ（またはスローダウン）異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        def time_stretch(x: np.ndarray, f: float) -> np.ndarray:
            """時系列データをf倍に時間伸縮する."""
            t_orig = len(x)
            original_time = np.arange(t_orig)
            new_t = int(t_orig / f)
            new_time = np.linspace(0, t_orig - 1, new_t)
            y = np.interp(new_time, original_time, x)
            return y

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            anomaly_segment = Y_window[loc_feature, anomaly_start:anomaly_end]
            freq = np.random.uniform(params['frequency'][0], params['frequency'][1]) # frequencyは範囲 [min, max]

            stretched_segment = time_stretch(anomaly_segment, freq)

            # 元の区間の長さを維持するように調整
            if len(stretched_segment) < len(anomaly_segment): # スピードアップ
                Y_temp[loc_feature, anomaly_start:anomaly_end] = np.pad(stretched_segment,
                                                                         (0, len(anomaly_segment) - len(stretched_segment)),
                                                                         mode='edge')
            elif len(stretched_segment) > len(anomaly_segment): # スローダウン
                Y_temp[loc_feature, anomaly_start:anomaly_end] = stretched_segment[:len(anomaly_segment)]
            else:
                Y_temp[loc_feature, anomaly_start:anomaly_end] = stretched_segment

            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

            # 平均値で補間 (元のコードの `Z_temp` 処理を反映)
            # これはZ_tempが元のデータであることを考えると奇妙だが、元のコードの挙動を再現
            # Z_temp[loc_feature, anomaly_start:anomaly_end] = np.mean(Z_temp[loc_feature, anomaly_start:anomaly_end])

        return Y_temp, Z_temp, mask_temp

    def _inject_noise(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウにノイズ異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            noise_amplitude = np.random.normal(loc=0, scale=params['scale'], size=anomaly_end - anomaly_start)
            Y_temp[loc_feature, anomaly_start:anomaly_end] += noise_amplitude
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_cutoff(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウにカットオフ（値を一定にする）異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            max_value = np.max(Y_window[loc_feature])
            min_value = np.min(Y_window[loc_feature])

            # ランダムな一定値に設定
            cutoff_value = np.random.uniform(low=min_value, high=max_value)
            Y_temp[loc_feature, anomaly_start:anomaly_end] = cutoff_value
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_average(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウに移動平均異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        def moving_average_with_padding(x: np.ndarray, w: int) -> np.ndarray:
            """端をエッジパディングして移動平均を計算する."""
            if len(x) == 0:
                return np.array([])

            # 修正案1: パディングを調整して長さを合わせる
            pad_left = w // 2
            pad_right = w - 1 - pad_left # wが偶数でも奇数でも、合計パディング幅が w-1 になる

            padded_x = np.pad(x, (pad_left, pad_right), mode='edge')

            return np.convolve(padded_x, np.ones(w) / w, 'valid')

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                if np.random.rand() > 0.5: # 前方から選択
                    anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                    anomaly_end = np.random.randint(low=anomaly_start + params['min_range'],
                                                    high=min(anomaly_start + params['anomaly_range'], window_size) + 1)
                else: # 後方から選択
                    anomaly_end = np.random.randint(low=params['min_range'], high=window_size + 1)
                    anomaly_start = np.random.randint(low=max(0, anomaly_end - params['anomaly_range']),
                                                      high=anomaly_end - params['min_range'] + 1)

            if anomaly_end <= anomaly_start:
                continue

            ma_segment = moving_average_with_padding(Y_window[loc_feature, anomaly_start:anomaly_end], params['ma_window'])
            Y_temp[loc_feature, anomaly_start:anomaly_end] = ma_segment
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_scale(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウにスケール変更異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            scale_factor = abs(np.random.normal(loc=1, scale=params['scale']))
            Y_temp[loc_feature, anomaly_start:anomaly_end] *= scale_factor
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_wander(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウにワンドル（徐々に変化する）異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            baseline = np.random.normal(loc=0, scale=params['scale'])

            # 線形に変化するベースラインを追加
            Y_temp[loc_feature, anomaly_start:anomaly_end] += np.linspace(0, baseline, anomaly_end - anomaly_start)
            # 異常区間以降もベースラインの影響を継続
            Y_temp[loc_feature, anomaly_end:] += baseline

            mask_temp[loc_feature, anomaly_start:] = 0 # 異常区間以降は全て異常とみなす
        return Y_temp, Z_temp, mask_temp

    def _inject_contextual(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウにコンテクスチュアル（文脈的）異常を注入します。
        これは、値自体は正常範囲内だが、文脈的に異常な変化を指します（ここでは線形変換）。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            a = np.random.normal(loc=1, scale=params['scale'])
            b = np.random.normal(loc=0, scale=params['scale'])

            # 線形変換を適用
            Y_temp[loc_feature, anomaly_start:anomaly_end] = a * Y_temp[loc_feature, anomaly_start:anomaly_end] + b
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_upsidedown(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウに上下反転異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            segment = Y_temp[loc_feature, anomaly_start:anomaly_end]
            mean = np.mean(segment)
            # 平均値を基準に反転
            Y_temp[loc_feature, anomaly_start:anomaly_end] = -(segment - mean) + mean
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_mixture(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ウィンドウに混合（他の時点のデータをコピー）異常を注入します。
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window)
        mask_temp = np.ones_like(Y_window, dtype=int)

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_features_list = np.random.choice(n_features, n_anom_features, replace=False)

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                anomaly_end = np.random.randint(low=anomaly_start + params['min_range'], high=window_size + 1)

            if anomaly_end <= anomaly_start:
                continue

            anomaly_length = anomaly_end - anomaly_start

            # データセット全体からランダムな区間をコピー
            # Y_window ではなく、元のデータセット Y からサンプリングするように修正
            # ただし、元のコードでは Y (dataset) 全体を使っているが、
            # _inject_mixtureのYはwindow_start:window_endのスライスなので注意。
            # ここではY_window (スライスされたデータ) の範囲内でコピーする。
            if window_size - anomaly_length <= 0: # コピー元が存在しない場合
                continue

            mixture_start = np.random.randint(low=0, high=window_size - anomaly_length + 1)
            mixture_end = mixture_start + anomaly_length

            Y_temp[loc_feature, anomaly_start:anomaly_end] = Y_window[loc_feature, mixture_start:mixture_end]
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    @staticmethod
    def _get_anomaly_dict(anomaly_types: List[str]) -> Dict[str, int]: # 型ヒントを追加
        anomaly_types = list(dict.fromkeys(anomaly_types))
        anomaly_dict = {}
        for i, anomaly_type in enumerate(anomaly_types):
            anomaly_dict[anomaly_type] = i
        return anomaly_dict

    @staticmethod
    def generate_one_hot(anomaly_types: List[str], anomaly_dict: Dict[str, int]) -> np.ndarray: # 型ヒントを追加
        '''
        input
        anomaly_types = ['normal','spike','cutoff']
        output
        one_hot_vectors = [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        '''
        labels = [anomaly_dict[atype] for atype in anomaly_types]
        # np.eye の引数は int である必要があるため、len(anomaly_dict) を明示的に int にキャスト
        one_hot_vectors = np.eye(int(len(anomaly_dict)))[labels]
        return one_hot_vectors

    @staticmethod
    def generate_anomaly_types(one_hot_vectors: Union[np.ndarray, t.Tensor], anomaly_dict: Dict[str, int]) -> List[str]: # 型ヒントを追加
        '''
        input
        one_hot_vectors = [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        output
        anomaly_types = ['normal','spike','cutoff']
        '''
        inverse_dict = {v: k for k, v in anomaly_dict.items()}

        # PyTorch Tensorの場合を考慮
        if isinstance(one_hot_vectors, t.Tensor):
            one_hot_vectors_np = one_hot_vectors.cpu().numpy()
        else:
            one_hot_vectors_np = one_hot_vectors

        if one_hot_vectors_np.ndim == 1:
            one_hot_vectors_np = one_hot_vectors_np[None, :] # [None, :]で次元を追加

        labels = np.argmax(one_hot_vectors_np, axis=1)
        return [inverse_dict[label] for label in labels]

class DataLoaderAug(AnomalyInjector):
    """
    データセットに異常を注入し、ウィンドウ単位でデータをロードするクラス。
    時系列データの前処理や、異常検知モデルのデータ拡張に利用できます。
    """

    def __init__(self,
                 dataset: np.ndarray,
                 anomaly_types: List[str],
                 params: Dict[str, Dict[str, Any]] = None,
                 window_size: int = None,
                 window_step: int = 1,
                 minmax: bool = False) -> None: # minmax引数を追加
        """
        DataLoaderAugのインスタンスを初期化します。

        Parameters
        ----------
        dataset : np.ndarray
            形状が [D, T] のデータセット。Dは特徴量数、Tは時間ステップ数。
        anomaly_types : List[str]
            注入する異常タイプのリスト (例: ['spike', 'noise', 'normal'])。
            'normal' を含めると、異常がない通常のウィンドウも生成されます。
            利用可能な異常タイプは `ANOMALY_TYPE_DEFAULTS` を参照。
            'random' を指定すると、定義済みの異常タイプからランダムに選択されます。
        params : Dict[str, Dict[str, Any]], optional
            各異常タイプに対するカスタムパラメータ。
            例: {'spike': {'scale': 2}, 'noise': {'scale': 0.05}}。
            指定しない場合、デフォルトパラメータが使用されます。
        window_size : int
            サンプリングするウィンドウのサイズ。
        window_step : int, optional
            ウィンドウ間のステップサイズ。デフォルトは1。
        minmax : bool, optional
            データセットを0-1にMin-Maxスケーリングし、最後に元のスケールに戻すかどうか。
            デフォルトはFalse。

        Attributes
        ----------
        Y_windows : torch.Tensor
            異常注入後のデータウィンドウの集合。形状: [B, D, W]
        Z_windows : torch.Tensor
            元のデータウィンドウの集合（異常なし）。形状: [B, D, W]
        anomaly_mask : torch.Tensor
            異常が注入された位置を示すマスク。0が異常箇所。形状: [B, D, W]
        label : torch.Tensor
            各ウィンドウの異常タイプを示すOne-Hotエンコーディングされたラベル。形状: [B, N_ANOMALY_TYPES]
        anomaly_dict : Dict[str, int]
            異常タイプ名とそれに対応するインデックスのマッピング辞書。
        original_min : np.ndarray
            Min-Maxスケーリング前の各特徴量の最小値。
        original_range : np.ndarray
            Min-Maxスケーリング前の各特徴量の値の範囲 (max - min)。
        """
        super().__init__()

        # Min-Maxスケーリングのための値を保存
        self.minmax = minmax
        if self.minmax:
            # 各特徴量ごとに最小値と範囲を計算（D, 1 の形状）
            self.original_min = np.min(dataset, axis=1, keepdims=True)
            self.original_range = np.max(dataset, axis=1, keepdims=True) - self.original_min
            # 範囲が0の場合（全ての値が同じ）は、0で割るエラーを避けるために小さな値を設定
            self.original_range[self.original_range == 0] = 1e-8

            # データセットを0-1にスケーリング
            self.dataset = (dataset - self.original_min) / self.original_range
        else:
            self.dataset = dataset # minmax=Falseの場合はそのまま

        self.window_size = window_size
        self.window_step = window_step
        self.anomaly_types = anomaly_types
        self.user_params = params if params is not None else {}

        all_unique_anomaly_types_for_dict = list(ANOMALY_TYPE_DEFAULTS.keys())
        if 'normal' not in all_unique_anomaly_types_for_dict:
            all_unique_anomaly_types_for_dict.insert(0, 'normal')
        if 'random' not in all_unique_anomaly_types_for_dict:
            all_unique_anomaly_types_for_dict.append('random')
        self.anomaly_dict = self._get_anomaly_dict(all_unique_anomaly_types_for_dict)

        self._validate_and_merge_parameters()

        self.Y_windows: List[t.Tensor] = []
        self.Z_windows: List[t.Tensor] = []
        self.anomaly_mask: List[t.Tensor] = []
        self.label: List[str] = []

        self._inject_anomalies_and_collect_windows()
        self._finalize_outputs()

    def _validate_and_merge_parameters(self) -> None:
        """
        初期化パラメータを検証し、デフォルト値とユーザー指定値をマージします。
        全ての既知の異常タイプに対して、共通パラメータとユーザー指定パラメータをマージして
        self.merged_params を構築します。
        """
        if not isinstance(self.dataset, np.ndarray) or self.dataset.ndim != 2:
            raise ValueError("`dataset` must be a 2D NumPy array with shape [D, T].")

        if self.window_size is None or not isinstance(self.window_size, int) or self.window_size <= 0:
            raise ValueError("`window_size` must be a positive integer.")

        if not isinstance(self.window_step, int) or self.window_step <= 0:
            raise ValueError("`window_step` must be a positive integer.")

        if self.dataset.shape[1] < self.window_size:
              raise ValueError(f"Dataset time dimension ({self.dataset.shape[1]}) is smaller than `window_size` ({self.window_size}).")

        self.merged_params: Dict[str, Dict[str, Any]] = {}

        all_known_anomaly_types = list(ANOMALY_TYPE_DEFAULTS.keys())
        if 'normal' not in all_known_anomaly_types:
            all_known_anomaly_types.insert(0, 'normal')
        if 'random' not in all_known_anomaly_types:
            all_known_anomaly_types.append('random')

        for anomaly_type_key in all_known_anomaly_types:
            default_for_current_type = {}
            if anomaly_type_key == 'normal':
                default_for_current_type = {}
            elif anomaly_type_key == 'random':
                default_for_current_type = {}
            else:
                default_for_current_type = ANOMALY_TYPE_DEFAULTS.get(anomaly_type_key, {})

            final_default_params = {**COMMON_ANOMALY_PARAMS, **default_for_current_type}
            user_params_for_key = self.user_params.get(anomaly_type_key, {})
            self.merged_params[anomaly_type_key] = {**final_default_params, **user_params_for_key}

        for requested_type in self.anomaly_types:
            if requested_type not in self.merged_params:
                raise ValueError(f"Unknown anomaly type specified in `anomaly_types`: '{requested_type}'. "
                                 f"Available types are: {list(ANOMALY_TYPE_DEFAULTS.keys()) + ['normal', 'random']}")

    def _inject_anomalies_and_collect_windows(self) -> None:
        """
        各異常タイプに基づき、データセットからウィンドウを抽出し、異常を注入します。
        """
        n_time = self.dataset.shape[1]

        for requested_anomaly_type in self.anomaly_types:
            current_windows_y: List[t.Tensor] = []
            current_windows_z: List[t.Tensor] = []
            current_masks: List[t.Tensor] = []

            window_start_idx = 0
            while window_start_idx + self.window_size <= n_time:
                current_data_slice = self.dataset[:, window_start_idx : window_start_idx + self.window_size]

                injected_anomaly_type = requested_anomaly_type
                if requested_anomaly_type == 'random':
                    available_anomaly_types_for_random = [
                        atype for atype in ANOMALY_TYPE_DEFAULTS.keys() if atype not in ['normal', 'random']
                    ]
                    if not available_anomaly_types_for_random:
                        raise RuntimeError("No anomaly types available for 'random' injection.")
                    injected_anomaly_type = np.random.choice(available_anomaly_types_for_random)

                if injected_anomaly_type == 'normal':
                    Y_win = np.copy(current_data_slice)
                    Z_win = np.copy(current_data_slice)
                    mask_win = np.ones_like(current_data_slice, dtype=int)
                else:
                    inject_method_name = f'_inject_{injected_anomaly_type}'
                    inject_method = getattr(self, inject_method_name, None)

                    if inject_method is None:
                        raise NotImplementedError(f"Anomaly injection method for '{injected_anomaly_type}' not found: {inject_method_name}")

                    params_for_type = self.merged_params.get(injected_anomaly_type, {})
                    Y_win, Z_win, mask_win = inject_method(current_data_slice, params_for_type)

                current_windows_y.append(t.from_numpy(Y_win).float())
                current_windows_z.append(t.from_numpy(Z_win).float())
                current_masks.append(t.from_numpy(mask_win).int())
                self.label.append(injected_anomaly_type)

                window_start_idx += self.window_step

            if current_windows_y:
                self.Y_windows.append(t.stack(current_windows_y, dim=0))
                self.Z_windows.append(t.stack(current_windows_z, dim=0))
                self.anomaly_mask.append(t.stack(current_masks, dim=0))

    def _finalize_outputs(self) -> None:
        """
        収集したウィンドウとラベルを最終的なPyTorch Tensorに結合し、
        必要に応じて元のスケールに戻します。
        """
        if not self.Y_windows:
            raise RuntimeError("No windows were generated. Check dataset size, window_size, and anomaly_types.")

        self.Y_windows = t.cat(self.Y_windows, dim=0)
        self.Z_windows = t.cat(self.Z_windows, dim=0)
        self.anomaly_mask = t.cat(self.anomaly_mask, dim=0)

        # Min-Maxスケーリングが有効な場合、元のスケールに戻す
        if self.minmax:
            # self.original_min と self.original_range は形状 [D, 1] なので、
            # PyTorchテンソルに変換して適切な次元でブロードキャストされるようにする
            original_min_tensor = t.from_numpy(self.original_min).float().unsqueeze(0) # [1, D, 1]
            original_range_tensor = t.from_numpy(self.original_range).float().unsqueeze(0) # [1, D, 1]

            self.Y_windows = self.Y_windows * original_range_tensor + original_min_tensor
            self.Z_windows = self.Z_windows * original_range_tensor + original_min_tensor


        self.label = t.from_numpy(self.generate_one_hot(self.label, self.anomaly_dict)).float()


class DataLoaderAugBatch(DataLoaderAug):
    """
    バッチ処理されたデータセットに異常を注入し、結果をバッチとして返すクラス。
    DataLoaderAugの機能を拡張し、既にウィンドウ化されたデータセットを処理します。
    """
    def __init__(self,
                 data: t.Tensor,
                 batch_size: int,
                 anomaly_types: List[str],
                 params: Dict[str, Dict[str, Any]] = None,
                 anomaly_types_for_dict: List[str] = None,
                 shuffle: bool = True,
                 minmax: bool = False) -> None: # minmax引数を追加
        """
        DataLoaderAugBatchのインスタンスを初期化します。

        Parameters
        ----------
        data : torch.Tensor
            形状が [B_in, D, T] の入力データ。B_in:元のバッチサイズ, D:特徴量, T:時間 (window_size)。
        batch_size : int
            このローダーが出力するバッチサイズ。
        anomaly_types : List[str]
            注入する異常タイプのリスト。
        params : Dict[str, Dict[str, Any]], optional
            各異常タイプに対するカスタムパラメータ。
        anomaly_types_for_dict : List[str], optional
            One-Hotエンコーディングの辞書を構築するために使用する全ての異常タイプ。
            指定しない場合、`anomaly_types`から辞書が作成されます。
            これにより、訓練セットとテストセットで一貫したラベル辞書を共有できます。
        shuffle : bool, optional
            結果のバッチをシャッフルするかどうか。デフォルトはTrue。
        minmax : bool, optional
            データセットを0-1にMin-Maxスケーリングし、最後に元のスケールに戻すかどうか。
            デフォルトはFalse。

        Attributes
        ----------
        Y_batches : torch.Tensor
            異常注入後のデータバッチの集合。形状: [B_out, D, T_window]
        Z_batches : torch.Tensor
            元のデータバッチの集合（異常なし）。形状: [B_out, D, T_window]
        anomaly_mask_batches : torch.Tensor
            異常が注入された位置を示すマスク。0が異常箇所。形状: [B_out, D, T_window]
        label_batches : torch.Tensor
            各バッチの異常タイプを示すOne-Hotエンコーディングされたラベル。形状: [B_out, N_ANOMALY_TYPES]
        anomaly_dict : Dict[str, int]
            異常タイプ名とそれに対応するインデックスのマッピング辞書。
        original_min_batch : np.ndarray
            DataLoaderAugBatchの入力データに対するMin-Maxスケーリング前の最小値。
        original_range_batch : np.ndarray
            DataLoaderAugBatchの入力データに対するMin-Maxスケーリング前の値の範囲。
        """
        if data.ndim != 3:
            raise ValueError("`data` must be a 3D Tensor with shape [Batch, Features, Time].")

        self.minmax = minmax
        self.data_input = data # オリジナルのPyTorch Tensorとして保持

        # DataLoaderAugBatchの入力データ自体をMin-Maxスケーリングする
        if self.minmax:
            # PyTorch TensorのままMin-Maxスケーリングを適用
            # 各特徴量ごとに最小値と範囲を計算（D, 1 の形状）
            # dataは [B, D, T] なので、D軸とT軸にわたって計算
            data_np = self.data_input.cpu().numpy() # NumPyに変換して計算
            self.original_min_batch = np.min(data_np, axis=(0, 2), keepdims=True) # [1, D, 1]
            self.original_range_batch = np.max(data_np, axis=(0, 2), keepdims=True) - self.original_min_batch
            self.original_range_batch[self.original_range_batch == 0] = 1e-8 # 0除算対策

            # データセットを0-1にスケーリング
            self.data_input = (self.data_input - t.from_numpy(self.original_min_batch).float()) / \
                              t.from_numpy(self.original_range_batch).float()

        # 親クラスのDataLoaderAugの__init__を呼び出し
        # DataLoaderAugBatchは既にウィンドウ化されたデータを扱うため、
        # 親クラスのdatasetはダミーで、minmax処理も親クラスでは行わない（このクラスで制御するため）
        dummy_dataset = np.zeros((self.data_input.shape[1], self.data_input.shape[2]))
        super().__init__(
            dataset=dummy_dataset,
            anomaly_types=anomaly_types,
            params=params,
            window_size=self.data_input.shape[2], # 入力データの最後の次元がwindow_size
            window_step=self.data_input.shape[2], # バッチ処理なので、ステップはウィンドウサイズと同じ
            minmax=False # DataLoaderAugBatchでminmaxを制御するため、親クラスでは無効化
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.anomaly_types_for_dict = anomaly_types_for_dict # 保持しておく

        # 親クラスのanomaly_dictは、このクラスのコンテキストでは一時的なものになる
        # 最終的なanomaly_dictは_finalize_outputs_batchで設定する

        self.Y_batches: List[t.Tensor] = []
        self.Z_batches: List[t.Tensor] = []
        self.anomaly_mask_batches: List[t.Tensor] = []
        self.label_batches: List[str] = []

        self._inject_anomalies_and_collect_batches()
        self._finalize_outputs_batch()

    def _inject_anomalies_and_collect_batches(self) -> None:
        """
        入力バッチデータに異常を注入し、結果を収集します。
        """
        num_input_batches = self.data_input.shape[0]

        for requested_anomaly_type in self.anomaly_types:
            injected_anomaly_type = requested_anomaly_type
            if requested_anomaly_type == 'random':
                available_anomaly_types_for_random = [
                    atype for atype in ANOMALY_TYPE_DEFAULTS.keys() if atype not in ['normal', 'random']
                ]
                if not available_anomaly_types_for_random:
                    raise RuntimeError("No anomaly types available for 'random' injection.")
                injected_anomaly_type = np.random.choice(available_anomaly_types_for_random)

            for i in range(num_input_batches):
                current_window_tensor = self.data_input[i] # [D, T]
                current_window_np = current_window_tensor.cpu().numpy() # NumPyに変換

                if injected_anomaly_type == 'normal':
                    Y_inj = np.copy(current_window_np)
                    Z_inj = np.copy(current_window_np)
                    mask_inj = np.ones_like(current_window_np, dtype=int)
                else:
                    inject_method_name = f'_inject_{injected_anomaly_type}'
                    inject_method = getattr(self, inject_method_name, None)
                    if inject_method is None:
                        raise NotImplementedError(f"Anomaly injection method for '{injected_anomaly_type}' not found: {inject_method_name}")

                    params_for_type = self.merged_params.get(injected_anomaly_type, {})
                    Y_inj, Z_inj, mask_inj = inject_method(current_window_np, params_for_type)

                self.Y_batches.append(t.from_numpy(Y_inj).float())
                self.Z_batches.append(t.from_numpy(Z_inj).float())
                self.anomaly_mask_batches.append(t.from_numpy(mask_inj).int())
                self.label_batches.append(injected_anomaly_type)

    def _finalize_outputs_batch(self) -> None:
        """
        収集したバッチとラベルを最終的なPyTorch Tensorに結合し、シャッフルします。
        必要に応じて元のスケールに戻します。
        """
        if not self.Y_batches:
            raise RuntimeError("No batches were generated. Check input data or anomaly_types.")

        self.Y_batches = t.stack(self.Y_batches, dim=0)
        self.Z_batches = t.stack(self.Z_batches, dim=0)
        self.anomaly_mask_batches = t.stack(self.anomaly_mask_batches, dim=0)

        # Min-Maxスケーリングが有効な場合、元のスケールに戻す
        if self.minmax:
            # original_min_batchとoriginal_range_batchは[1, D, 1]のNumPy配列
            # これらをPyTorchテンソルに変換してブロードキャスト可能にする
            original_min_tensor = t.from_numpy(self.original_min_batch).float() # [1, D, 1]
            original_range_tensor = t.from_numpy(self.original_range_batch).float() # [1, D, 1]

            # Y_batches, Z_batches は [B, D, W] 形状なので、
            # original_min_tensorとoriginal_range_tensorはD軸でブロードキャストされる
            self.Y_batches = self.Y_batches * original_range_tensor + original_min_tensor
            self.Z_batches = self.Z_batches * original_range_tensor + original_min_tensor

        # ラベルのOne-Hotエンコーディング
        # anomaly_types_for_dict が指定されていればそれを使用
        if self.anomaly_types_for_dict:
            final_anomaly_dict = self._get_anomaly_dict(self.anomaly_types_for_dict)
        else:
            # 指定がなければ、実際に生成されたユニークな anomaly_type を基に辞書を作成
            final_anomaly_dict = self._get_anomaly_dict(list(dict.fromkeys(self.label_batches)))

        # 最終的な anomaly_dict をインスタンス変数として保存しておく
        self.anomaly_dict = final_anomaly_dict

        self.label_batches = t.from_numpy(self.generate_one_hot(self.label_batches, self.anomaly_dict)).float()

        if self.shuffle:
            permutation = t.randperm(self.Y_batches.shape[0])
            self.Y_batches = self.Y_batches[permutation]
            self.Z_batches = self.Z_batches[permutation]
            self.anomaly_mask_batches = self.anomaly_mask_batches[permutation]
            self.label_batches = self.label_batches[permutation]

    def __len__(self) -> int:
        """バッチの総数を返します。"""
        return (self.Y_batches.shape[0] + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        """指定されたインデックスのバッチを返します。"""
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.Y_batches.shape[0])
        return (self.Y_batches[start:end],
                self.Z_batches[start:end],
                self.anomaly_mask_batches[start:end],
                self.label_batches[start:end])
