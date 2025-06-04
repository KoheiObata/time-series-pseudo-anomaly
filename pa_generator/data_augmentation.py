import numpy as np
import torch as t
from typing import Dict, List, Any, Tuple, Union

# --- Default parameter definition per anomaly type ---
# Default parameters common to all anomaly types
COMMON_ANOMALY_PARAMS: Dict[str, Any] = {
    'min_range': 20,
    'min_features': 1,
    'max_features': 5,
}

# Default parameters specific to each anomaly type.
# Common parameters are overwritten or added here.
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
    Base class for injecting various types of anomalies into time series data.
    Each anomaly injection method (e.g., _inject_spike) is implemented in this class
    or its subclasses.
    """
    def __init__(self) -> None:
        # This class is not intended to be instantiated directly.
        # It exists for subclasses to inherit specific parameters.
        pass

    def _inject_spike(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects a spike anomaly into the window.

        Parameters
        ----------
        Y_window : np.ndarray
            Window data with shape [D, T]. D: features, T: time.
        params : Dict[str, Any]
            Parameters for the spike anomaly (e.g., 'min_features', 'max_features', 'scale').

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Data after anomaly injection, original data, anomaly mask.
        """
        n_features, window_size = Y_window.shape
        max_features = min(params['max_features'], n_features)

        Y_temp = np.copy(Y_window)
        Z_temp = np.copy(Y_window) # Original data
        mask_temp = np.ones_like(Y_window, dtype=int) # Anomaly locations are 0

        if params['min_features'] == max_features:
            n_anom_features = max_features
        else:
            n_anom_features = np.random.randint(low=params['min_features'], high=max_features + 1)

        loc_time = np.random.randint(low=0, high=window_size, size=n_anom_features)
        loc_features = np.random.randint(low=0, high=n_features, size=n_anom_features)

        # Add spike
        Y_temp[loc_features, loc_time] += np.random.normal(loc=0, scale=params['scale'], size=n_anom_features)

        # Update mask
        mask_temp[loc_features, loc_time] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_flip(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects a flip anomaly into the window.
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
                if np.random.rand() > 0.5: # Select from front
                    anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                    anomaly_end = np.random.randint(low=anomaly_start + params['min_range'],
                                                     high=min(anomaly_start + params['anomaly_range'], window_size) + 1)
                else: # Select from back
                    anomaly_end = np.random.randint(low=params['min_range'], high=window_size + 1)
                    anomaly_start = np.random.randint(low=max(0, anomaly_end - params['anomaly_range']),
                                                      high=anomaly_end - params['min_range'] + 1)

            # Skip if range is invalid (shouldn't happen but just in case)
            if anomaly_end <= anomaly_start:
                continue

            # Flip the sequence
            Y_temp[loc_feature, anomaly_start:anomaly_end] = Y_temp[loc_feature, anomaly_start:anomaly_end][::-1]
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

        return Y_temp, Z_temp, mask_temp

    def _inject_speedup(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects a speed-up (or slow-down) anomaly into the window.
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
            """Time stretches time series data by a factor of f."""
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

            # Skip if range is invalid (shouldn't happen but just in case)
            if anomaly_end <= anomaly_start:
                continue

            anomaly_segment = Y_window[loc_feature, anomaly_start:anomaly_end]
            freq = np.random.uniform(params['frequency'][0], params['frequency'][1]) # frequency is range [min, max]

            stretched_segment = time_stretch(anomaly_segment, freq)

            # Adjust to maintain the length of the original segment
            if len(stretched_segment) < len(anomaly_segment): # Speed up
                Y_temp[loc_feature, anomaly_start:anomaly_end] = np.pad(stretched_segment,
                                                                          (0, len(anomaly_segment) - len(stretched_segment)),
                                                                          mode='edge')
            elif len(stretched_segment) > len(anomaly_segment): # Slow down
                Y_temp[loc_feature, anomaly_start:anomaly_end] = stretched_segment[:len(anomaly_segment)]
            else:
                Y_temp[loc_feature, anomaly_start:anomaly_end] = stretched_segment

            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0

            # Interpolate with mean (reflects `Z_temp` processing in original code)
            # This is strange considering Z_temp is original data, but reproduces original code behavior
            # Z_temp[loc_feature, anomaly_start:anomaly_end] = np.mean(Z_temp[loc_feature, anomaly_start:anomaly_end])

        return Y_temp, Z_temp, mask_temp

    def _inject_noise(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects a noise anomaly into the window.
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
        Injects a cutoff (constant value) anomaly into the window.
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

            # Set to a random constant value
            cutoff_value = np.random.uniform(low=min_value, high=max_value)
            Y_temp[loc_feature, anomaly_start:anomaly_end] = cutoff_value
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_average(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects a moving average anomaly into the window.
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
            """Calculates moving average with edge padding at the ends."""
            if len(x) == 0:
                return np.array([])

            # Revision 1: Adjust padding to match length
            pad_left = w // 2
            pad_right = w - 1 - pad_left # Total padding width is w-1, whether w is even or odd

            padded_x = np.pad(x, (pad_left, pad_right), mode='edge')

            return np.convolve(padded_x, np.ones(w) / w, 'valid')

        for loc_feature in loc_features_list:
            if params['min_range'] == window_size:
                anomaly_start = 0
                anomaly_end = window_size
            else:
                if np.random.rand() > 0.5: # Select from front
                    anomaly_start = np.random.randint(low=0, high=window_size - params['min_range'] + 1)
                    anomaly_end = np.random.randint(low=anomaly_start + params['min_range'],
                                                     high=min(anomaly_start + params['anomaly_range'], window_size) + 1)
                else: # Select from back
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
        Injects a scale change anomaly into the window.
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
        Injects a wander (gradual change) anomaly into the window.
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

            # Add a linearly changing baseline
            Y_temp[loc_feature, anomaly_start:anomaly_end] += np.linspace(0, baseline, anomaly_end - anomaly_start)
            # Continue the baseline effect after the anomaly segment
            Y_temp[loc_feature, anomaly_end:] += baseline

            mask_temp[loc_feature, anomaly_start:] = 0 # Mark everything from anomaly start as anomalous
        return Y_temp, Z_temp, mask_temp

    def _inject_contextual(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects a contextual anomaly into the window.
        This refers to changes that are within the normal range of values, but are
        anomalous in context (here, a linear transformation).
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

            # Apply linear transformation
            Y_temp[loc_feature, anomaly_start:anomaly_end] = a * Y_temp[loc_feature, anomaly_start:anomaly_end] + b
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_upsidedown(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects an upside-down anomaly into the window.
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
            # Flip based on the mean
            Y_temp[loc_feature, anomaly_start:anomaly_end] = -(segment - mean) + mean
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    def _inject_mixture(self, Y_window: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Injects a mixture (copy data from another point in time) anomaly into the window.
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

            # Copy a random segment from the entire dataset
            # Note that the original code uses the entire Y (dataset),
            # but _inject_mixture's Y is a slice of window_start:window_end.
            # Here, we copy within the Y_window (sliced data) range.
            if window_size - anomaly_length <= 0: # If no source to copy from
                continue

            mixture_start = np.random.randint(low=0, high=window_size - anomaly_length + 1)
            mixture_end = mixture_start + anomaly_length

            Y_temp[loc_feature, anomaly_start:anomaly_end] = Y_window[loc_feature, mixture_start:mixture_end]
            mask_temp[loc_feature, anomaly_start:anomaly_end] = 0
        return Y_temp, Z_temp, mask_temp

    @staticmethod
    def _get_anomaly_dict(anomaly_types: List[str]) -> Dict[str, int]: # Added type hint
        anomaly_types = list(dict.fromkeys(anomaly_types))
        anomaly_dict = {}
        for i, anomaly_type in enumerate(anomaly_types):
            anomaly_dict[anomaly_type] = i
        return anomaly_dict

    @staticmethod
    def generate_one_hot(anomaly_types: List[str], anomaly_dict: Dict[str, int]) -> np.ndarray: # Added type hint
        '''
        input
        anomaly_types = ['normal','spike','cutoff']
        output
        one_hot_vectors = [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        '''
        labels = [anomaly_dict[atype] for atype in anomaly_types]
        # np.eye's argument must be an int, so explicitly cast len(anomaly_dict) to int
        one_hot_vectors = np.eye(int(len(anomaly_dict)))[labels]
        return one_hot_vectors

    @staticmethod
    def generate_anomaly_types(one_hot_vectors: Union[np.ndarray, t.Tensor], anomaly_dict: Dict[str, int]) -> List[str]: # Added type hint
        '''
        input
        one_hot_vectors = [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
                            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
        output
        anomaly_types = ['normal','spike','cutoff']
        '''
        inverse_dict = {v: k for k, v in anomaly_dict.items()}

        # Consider PyTorch Tensor case
        if isinstance(one_hot_vectors, t.Tensor):
            one_hot_vectors_np = one_hot_vectors.cpu().numpy()
        else:
            one_hot_vectors_np = one_hot_vectors

        if one_hot_vectors_np.ndim == 1:
            one_hot_vectors_np = one_hot_vectors_np[None, :] # Add dimension with [None, :]

        labels = np.argmax(one_hot_vectors_np, axis=1)
        return [inverse_dict[label] for label in labels]

class DataLoaderAug(AnomalyInjector):
    """
    A class for injecting anomalies into a dataset and loading data in windows.
    Can be used for time series data preprocessing or data augmentation for anomaly detection models.
    """

    def __init__(self,
                 dataset: np.ndarray,
                 anomaly_types: List[str],
                 params: Dict[str, Dict[str, Any]] = None,
                 window_size: int = None,
                 window_step: int = 1,
                 minmax: bool = False) -> None: # Added minmax argument
        """
        Initializes an instance of DataLoaderAug.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset with shape [D, T]. D is number of features, T is number of time steps.
        anomaly_types : List[str]
            List of anomaly types to inject (e.g., ['spike', 'noise', 'normal']).
            Including 'normal' will generate normal windows without anomalies.
            Available anomaly types can be found in `ANOMALY_TYPE_DEFAULTS`.
            Specifying 'random' will randomly select from predefined anomaly types.
        params : Dict[str, Dict[str, Any]], optional
            Custom parameters for each anomaly type.
            Example: {'spike': {'scale': 2}, 'noise': {'scale': 0.05}}.
            If not specified, default parameters will be used.
        window_size : int
            Size of the window to sample.
        window_step : int, optional
            Step size between windows. Default is 1.
        minmax : bool, optional
            Whether to Min-Max scale the dataset to 0-1 and then revert to original scale at the end.
            Default is False.

        Attributes
        ----------
        Y_windows : torch.Tensor
            Collection of data windows after anomaly injection. Shape: [B, D, W]
        Z_windows : torch.Tensor
            Collection of original data windows (without anomalies). Shape: [B, D, W]
        anomaly_mask : torch.Tensor
            Mask indicating locations where anomalies were injected. 0 means anomaly location. Shape: [B, D, W]
        label : torch.Tensor
            One-Hot encoded labels indicating the anomaly type of each window. Shape: [B, N_ANOMALY_TYPES]
        anomaly_dict : Dict[str, int]
            Mapping dictionary between anomaly type names and their corresponding indices.
        original_min : np.ndarray
            Minimum value of each feature before Min-Max scaling.
        original_range : np.ndarray
            Range of values (max - min) for each feature before Min-Max scaling.
        """
        super().__init__()

        # Save values for Min-Max scaling
        self.minmax = minmax
        if self.minmax:
            # Calculate min and range for each feature (shape D, 1)
            self.original_min = np.min(dataset, axis=1, keepdims=True)
            self.original_range = np.max(dataset, axis=1, keepdims=True) - self.original_min
            # If range is 0 (all values are the same), set a small value to avoid division by zero error
            self.original_range[self.original_range == 0] = 1e-8

            # Scale dataset to 0-1
            self.dataset = (dataset - self.original_min) / self.original_range
        else:
            self.dataset = dataset # If minmax=False, use as is

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
        Validates initialization parameters and merges default and user-specified values.
        Constructs self.merged_params by merging common parameters and user-specified parameters
        for all known anomaly types.
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
        Extracts windows from the dataset and injects anomalies based on each anomaly type.
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
        Combines the collected windows and labels into final PyTorch Tensors,
        and reverts to the original scale if necessary.
        """
        if not self.Y_windows:
            raise RuntimeError("No windows were generated. Check dataset size, window_size, and anomaly_types.")

        self.Y_windows = t.cat(self.Y_windows, dim=0)
        self.Z_windows = t.cat(self.Z_windows, dim=0)
        self.anomaly_mask = t.cat(self.anomaly_mask, dim=0)

        # If Min-Max scaling is enabled, revert to original scale
        if self.minmax:
            # self.original_min and self.original_range have shape [D, 1], so
            # convert them to PyTorch tensors and ensure they broadcast correctly across dimensions
            original_min_tensor = t.from_numpy(self.original_min).float().unsqueeze(0) # [1, D, 1]
            original_range_tensor = t.from_numpy(self.original_range).float().unsqueeze(0) # [1, D, 1]

            self.Y_windows = self.Y_windows * original_range_tensor + original_min_tensor
            self.Z_windows = self.Z_windows * original_range_tensor + original_min_tensor


        self.label = t.from_numpy(self.generate_one_hot(self.label, self.anomaly_dict)).float()


class DataLoaderAugBatch(DataLoaderAug):
    """
    A class that injects anomalies into a batched dataset and returns the results as batches.
    Extends DataLoaderAug to process already-windowed data.
    """
    def __init__(self,
                 dataset: np.ndarray,
                 anomaly_types: List[str],
                 params: Dict[str, Dict[str, Any]] = None,
                 minmax: bool = False) -> None: # Added minmax argument
        """
        Initializes an instance of DataLoaderAugBatch.

        Parameters
        ----------
        dataset : torch.Tensor
            Input data with shape [B_in, D, T]. B_in: original batch size, D: features, T: time (window_size).
        anomaly_types : List[str]
            List of anomaly types to inject.
        params : Dict[str, Dict[str, Any]], optional
            Custom parameters for each anomaly type.
        minmax : bool, optional
            Whether to Min-Max scale the dataset to 0-1 and then revert to original scale at the end.
            Default is False.

        Attributes
        ----------
        Y_batches : torch.Tensor
            Collection of data batches after anomaly injection. Shape: [B_out, D, T_window]
        Z_batches : torch.Tensor
            Collection of original data batches (without anomalies). Shape: [B_out, D, T_window]
        anomaly_mask_batches : torch.Tensor
            Mask indicating locations where anomalies were injected. 0 means anomaly location. Shape: [B_out, D, T_window]
        label_batches : torch.Tensor
            One-Hot encoded labels indicating the anomaly type of each batch. Shape: [B_out, N_ANOMALY_TYPES]
        anomaly_dict : Dict[str, int]
            Mapping dictionary between anomaly type names and their corresponding indices.
        original_min_batch : np.ndarray
            Minimum value of the input data for DataLoaderAugBatch before Min-Max scaling.
        original_range_batch : np.ndarray
            Range of values for the input data for DataLoaderAugBatch before Min-Max scaling.
        """
        if dataset.ndim != 3:
            raise ValueError("`data` must be a 3D Tensor with shape [Batch, Features, Time].")

        self.minmax = minmax
        self.data_input = dataset # Keep as original PyTorch Tensor

        # Min-Max scale the input data for DataLoaderAugBatch itself
        if self.minmax:
            # Apply Min-Max scaling directly to PyTorch Tensor
            # Calculate min and range for each feature (shape D, 1)
            # data is [B, D, T], so calculate across B and T axes
            self.original_min_batch = np.min(dataset, axis=(0, 2), keepdims=True) # [1, D, 1]
            self.original_range_batch = np.max(dataset, axis=(0, 2), keepdims=True) - self.original_min_batch
            self.original_range_batch[self.original_range_batch == 0] = 1e-8 # Handle division by zero

            # Scale dataset to 0-1
            self.dataset = (self.data_input - self.original_min_batch) / self.original_range_batch
        else:
            self.dataset = self.data_input

        # Call the __init__ of the parent class DataLoaderAug
        # DataLoaderAugBatch handles already windowed data, so
        # the parent class's dataset is a dummy, and minmax processing is handled here, not by the parent class.
        dummy_dataset = np.zeros((self.data_input.shape[1], self.data_input.shape[2]))
        super().__init__(
            dataset=dummy_dataset,
            anomaly_types=anomaly_types,
            params=params,
            window_size=self.data_input.shape[2], # The last dimension of input data is window_size
            window_step=self.data_input.shape[2], # For batch processing, step is same as window size
            minmax=False # Disable minmax in parent class as it's controlled by DataLoaderAugBatch
        )

        # The parent class's anomaly_dict will be temporary in this class's context
        # The final anomaly_dict will be set in _finalize_outputs_batch

        self.Y_batches: List[t.Tensor] = []
        self.Z_batches: List[t.Tensor] = []
        self.anomaly_mask_batches: List[t.Tensor] = []
        self.label_batches: List[str] = []

        self._inject_anomalies_and_collect_batches()
        self._finalize_outputs_batch()

    def _inject_anomalies_and_collect_batches(self) -> None:
        """
        Injects anomalies into the input batch data and collects the results.
        """
        num_input_batches = self.data_input.shape[0]

        for requested_anomaly_type in self.anomaly_types:
            for i in range(num_input_batches):
                current_window_np = self.data_input[i] # [D, T]

                injected_anomaly_type = requested_anomaly_type
                if requested_anomaly_type == 'random':
                    available_anomaly_types_for_random = [
                        atype for atype in ANOMALY_TYPE_DEFAULTS.keys() if atype not in ['normal', 'random']
                    ]
                    if not available_anomaly_types_for_random:
                        raise RuntimeError("No anomaly types available for 'random' injection.")
                    injected_anomaly_type = str(np.random.choice(available_anomaly_types_for_random))

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
        Combines the collected batches and labels into final PyTorch Tensors and shuffles them.
        Reverts to original scale if necessary.
        """
        if not self.Y_batches:
            raise RuntimeError("No batches were generated. Check input data or anomaly_types.")

        self.Y_batches = t.stack(self.Y_batches, dim=0)
        self.Z_batches = t.stack(self.Z_batches, dim=0)
        self.anomaly_mask_batches = t.stack(self.anomaly_mask_batches, dim=0)

        # If Min-Max scaling is enabled, revert to original scale
        if self.minmax:
            # original_min_batch and original_range_batch are NumPy arrays of shape [1, D, 1]
            # Convert them to PyTorch tensors to enable broadcasting
            original_min_tensor = t.from_numpy(self.original_min_batch).float() # [1, D, 1]
            original_range_tensor = t.from_numpy(self.original_range_batch).float() # [1, D, 1]

            # Y_batches, Z_batches have shape [B, D, W], so
            # original_min_tensor and original_range_tensor will broadcast along the D-axis
            self.Y_batches = self.Y_batches * original_range_tensor + original_min_tensor
            self.Z_batches = self.Z_batches * original_range_tensor + original_min_tensor

        # One-Hot encode labels
        # Create a dictionary based on the actually generated unique anomaly_types
        final_anomaly_dict = self._get_anomaly_dict(list(dict.fromkeys(self.label_batches)))

        # Save the final anomaly_dict as an instance variable
        self.anomaly_dict = final_anomaly_dict

        self.label_batches = t.from_numpy(self.generate_one_hot(self.label_batches, self.anomaly_dict)).float()

