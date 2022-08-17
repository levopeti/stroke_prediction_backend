import pandas as pd
import numpy as np
from termcolor import colored
from random import randint


class Measurement(object):
    def __init__(self, measurement_name, synchronizing=True):
        self.measurement_name = measurement_name
        self.synchronizing = synchronizing

        self.log_list = list()

        self.measurement_dict = {
            ("left", "arm", "acc"): None,
            ("left", "arm", "gyr"): None,
            ("left", "leg", "acc"): None,
            ("left", "leg", "gyr"): None,
            ("right", "arm", "acc"): None,
            ("right", "arm", "gyr"): None,
            ("right", "leg", "acc"): None,
            ("right", "leg", "gyr"): None,
        }

        self.diff_dict = {
            ("left", "arm", "acc"): None,
            ("left", "arm", "gyr"): None,
            ("left", "leg", "acc"): None,
            ("left", "leg", "gyr"): None,
            ("right", "arm", "acc"): None,
            ("right", "arm", "gyr"): None,
            ("right", "leg", "acc"): None,
            ("right", "leg", "gyr"): None,
        }

    # #### checks for measurement ####
    def check_frequency(self, expected_delta, eps):
        for df in self.measurement_dict.values():
            if df is not None:
                time_stamps = df["epoch"].values
                deltas = np.diff(time_stamps)
                if np.any(deltas < expected_delta - eps) or np.any(deltas > expected_delta + eps):
                    self.log_list.append(colored("frequency is not correct,"
                                                 " min: {}, max: {}, avg: {}".format(np.min(deltas),
                                                                                     np.max(deltas),
                                                                                     np.mean(deltas)), "red"))

    def print_log(self):
        for log in self.log_list:
            print(log)

    def get_measurement_df(self, key, only_valid=True):
        columns_key_dict = {"acc": ("epoch (ms)", "x-axis (g)", "y-axis (g)", "z-axis (g)"),
                            "gyr": ("epoch (ms)", "x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)")}

        def cut_valid_part(_meas_df):
            if only_valid and self.valid_start_time is not None and self.valid_end_time is not None:
                # _meas_df = _meas_df[_meas_df["epoch"] > self.valid_start_time.timestamp() * 1000]
                # _meas_df = _meas_df[_meas_df["epoch"] < self.valid_end_time.timestamp() * 1000]
                _meas_df = _meas_df[_meas_df["epoch"] > int(self.valid_start_time / np.timedelta64(1, 'ms'))]
                _meas_df = _meas_df[_meas_df["epoch"] < int(self.valid_end_time / np.timedelta64(1, 'ms'))]
            return _meas_df

        def cut_for_mutual_part(_measurement_dict):
            min_ts = 0
            max_ts = float('inf')

            for meas in _measurement_dict.values():
                if meas["epoch"].min() > min_ts:
                    min_ts = meas["epoch"].min()

                if meas["epoch"].max() < max_ts:
                    max_ts = meas["epoch"].max()

            for _k, meas in _measurement_dict.items():
                # print(len(meas[(meas["epoch"] >= min_ts) & (meas["epoch"] <= max_ts)]))
                _measurement_dict[_k] = meas[(meas["epoch"] >= min_ts) & (meas["epoch"] <= max_ts)]

            return _measurement_dict

        def synchronize(_measurement_dict):
            _measurement_dict = cut_for_mutual_part(_measurement_dict)

            base_df = None
            for _k, _df in _measurement_dict.items():
                if base_df is None:
                    base_df = _df.sort_values('epoch')
                else:
                    _df = _df.sort_values('epoch')
                    merged_df = pd.merge_asof(base_df, _df, on="epoch", tolerance=40, direction='nearest')
                    assert merged_df.isna().sum().sum() == 0

                    columns_for_drop = list()
                    for c_name in merged_df.columns:
                        if c_name.find("_x") != -1:
                            columns_for_drop.append(c_name)
                    merged_df.drop(columns_for_drop, inplace=True, axis=1)

                    # remove _y
                    for c_name in merged_df.columns:
                        if c_name.find("_y") != -1:
                            merged_df.rename(columns={c_name: c_name[:-2]}, inplace=True)

                    _measurement_dict[_k] = merged_df

            for _k, _df in _measurement_dict.items():
                if len(_df) == 0:
                    print(colored("zero length of data {}, {}".format(self.measurement_name, _k), "red"))
            return _measurement_dict

        if not self.lightweight and self.measurement_dict[key] is not None:
            meas_df = self.measurement_dict[key]
        else:
            if self.synchronizing:
                tmp_measurement_dict = dict()
                for k in self.measurement_path_dict.keys():
                    meas_df = read_csv(k)
                    tmp_measurement_dict[k] = meas_df

                tmp_measurement_dict = synchronize(tmp_measurement_dict)
                meas_df = tmp_measurement_dict[key]

                if not self.lightweight:
                    self.measurement_dict = tmp_measurement_dict
            else:
                meas_df = read_csv(key)
                if not self.lightweight:
                    self.measurement_dict[key] = meas_df

        assert len(meas_df) > 0
        # if len(meas_df) == 0:
        #     print(self.measurement_name)
        #     print(key)
        #     exit()

        return meas_df

    def get_all_measurements_df(self, only_valid=True):
        result_dict = dict()

        for k in self.measurement_path_dict.keys():
            result_dict[k] = self.get_measurement_df(k, only_valid=only_valid)

        return result_dict

    def get_mutual_limb_masks(self, limb, meas_type="acc", only_valid=True):
        left_meas = self.get_measurement_df(("left", limb, meas_type), only_valid)
        right_meas = self.get_measurement_df(("right", limb, meas_type), only_valid)

        left_mask = (left_meas["epoch"] >= right_meas["epoch"].min()) & \
                    (left_meas["epoch"] <= right_meas["epoch"].max())
        right_mask = (right_meas["epoch"] >= left_meas["epoch"].min()) & \
                     (right_meas["epoch"] <= left_meas["epoch"].max())

        return left_mask, right_mask

    def calculate_diff(self, key, use_abs=True, only_valid=True):
        if not self.lightweight and self.diff_dict[key] is not None:
            result = self.diff_dict[key]
        else:
            meas_type = key[2]
            meas = self.get_measurement_df(key, only_valid=only_valid)
            x_y_z = [meas[("x-axis", "y-axis", "z-axis")[i]] for i in range(3)]

            if meas_type == "acc":
                x_diff, y_diff, z_diff = [np.diff(m) for m in x_y_z]
            else:
                x_diff, y_diff, z_diff = [m.values for m in x_y_z]

            if use_abs:
                result = np.abs(x_diff) + np.abs(y_diff) + np.abs(z_diff)
            else:
                result = x_diff + y_diff + z_diff

            if not self.lightweight:
                self.diff_dict[key] = result

        assert len(result) > 0
        return result

    def get_diff(self, key, length=None, start_idx=None, use_abs=True, only_valid=True, mask=None):
        result = self.calculate_diff(key, use_abs, only_valid)

        if mask is not None:
            try:
                result = result[mask[:len(result)]]
            except Exception as e:
                print(e)
                print(key, length, start_idx)
                print(result.shape)
                print(mask[:len(result)].shape)
                exit()

        if length is not None:
            assert length < len(result)

            start_idx = start_idx if start_idx is not None else randint(0, len(result) - (length + 1))
            if start_idx > len(result) - (length + 1):
                raise ValueError("start_idx is too large")
            result = result[start_idx:start_idx + length]

        assert len(result) > 0
        return result

    def sweep_diff(self, key, length, mean=False):
        result_list = list()
        start_idx = 0

        while True:
            try:
                if mean:
                    result_list.append(self.get_diff(key, length, start_idx).mean())
                else:
                    result_list.append(self.get_diff(key, length, start_idx))
            except ValueError:
                break
            start_idx = start_idx + 1

        return result_list

    def get_all_diff(self, only_valid=True):
        result_dict = dict()

        for k in self.measurement_path_dict.keys():
            result_dict[k] = self.get_diff(k, only_valid=only_valid)

        return result_dict

    def get_limb_diff_mean(self, limb, meas_type="acc", length=None, start_idx=None, use_abs=True, only_valid=True):
        assert limb in ["arm", "leg"]
        assert meas_type in ["acc", "gyr"]
        left_key, right_key = ("left", limb, meas_type), ("right", limb, meas_type)

        if not self.synchronizing:
            left_mask, right_mask = self.get_mutual_limb_masks(limb, meas_type)
        else:
            left_mask, right_mask = None, None

        left_diff = self.get_diff(left_key, length, start_idx, use_abs, only_valid, left_mask)
        right_diff = self.get_diff(right_key, length, start_idx, use_abs, only_valid, right_mask)

        result = np.abs(left_diff.mean() - right_diff.mean())
        is_five = self.class_value_dict[("left", limb)] == 5 or self.class_value_dict[("right", limb)] == 5
        return result, is_five

    def get_limb_ratio_mean(self, limb, meas_type="acc", length=None, start_idx=None, use_abs=True, only_valid=True,
                            mean_first=True):
        assert limb in ["arm", "leg"]
        assert meas_type in ["acc", "gyr"]
        left_key, right_key = ("left", limb, meas_type), ("right", limb, meas_type)

        if not self.synchronizing:
            left_mask, right_mask = self.get_mutual_limb_masks(limb, meas_type)
        else:
            left_mask, right_mask = None, None

        left_diff = self.get_diff(left_key, length, start_idx, use_abs, only_valid, left_mask)
        right_diff = self.get_diff(right_key, length, start_idx, use_abs, only_valid, right_mask)

        if mean_first:
            if self.class_value_dict[("left", limb)] > self.class_value_dict[("right", limb)]:
                result = left_diff.sum() / right_diff.sum()
            else:
                result = right_diff.sum() / left_diff.sum()
        else:
            left_diff = left_diff + 0.1
            right_diff = right_diff + 0.1
            if self.class_value_dict[("left", limb)] > self.class_value_dict[("right", limb)]:
                result = np.mean(left_diff / right_diff)
            else:
                result = np.mean(right_diff / left_diff)

        is_five = self.class_value_dict[("left", limb)] == 5 or self.class_value_dict[("right", limb)] == 5
        return result, is_five
