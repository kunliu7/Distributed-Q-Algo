import datetime
import os
import pickle

import pandas as pd


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%m%d-%H%M%S")


class BaseDataMgr():
    file_type = ""
    data_dir = ""  # the dir in the ./data folder
    load_param = {}
    save_param = {}

    def __init__(self) -> None:
        cur_file_dir = os.path.dirname(__file__)
        path_to_data_folder = os.path.join(cur_file_dir, "../../")
        self.data_dir = os.path.join(path_to_data_folder, self.data_dir)

    def load(self, *args, **kwargs):
        filepath = self.get_filepath(*args, **kwargs)
        if self.file_type == "pkl":
            with open(filepath, "rb") as f:
                data = pickle.load(f, **self.load_param)
        elif self.file_type == "csv":
            data = pd.read_csv(filepath, **self.load_param)
        else:
            raise ValueError(f"Unknown file type {self.file_type}")

        return data

    def save(self, data, *args, **kwargs):
        filepath = self.get_filepath(*args, **kwargs)
        if self.file_type == "pkl":
            with open(filepath, "wb") as f:
                pickle.dump(data, f, **self.save_param)
        elif self.file_type == "csv":
            data.to_csv(filepath, index=False, **self.save_param)
        else:
            raise ValueError(f"Unknown file type {self.file_type}")

    def is_data_exist(self, *args, **kwargs) -> bool:
        filepath = self.get_filepath(*args, **kwargs)
        return os.path.exists(filepath)

    def get_signature(self, *args, **kwargs) -> str:
        sign = ""
        for k, w in kwargs.items():
            if isinstance(w, list):
                w = ",".join(map(str, w))
            sign += f"{str(k)}={str(w)}-"

        sign = sign[:-1]
        return sign

    def get_filepath(self, *args, **kwargs) -> str:
        sign = self.get_signature(*args, **kwargs)
        filepath = os.path.join(self.data_dir, f"{sign}.{self.file_type}")
        return filepath

    def get_data_dir(self, *args, **kwargs) -> str:
        return self.data_dir


class NISQFanoutDataMgr(BaseDataMgr):
    data_dir = "data/nisq/fanout"
    file_type = "pkl"

class NISQCswapDataMgr(BaseDataMgr):
    data_dir = "data/nisq/cswap"
    file_type = "csv"

class NISQTeleportDataMgr(BaseDataMgr):
    data_dir = "data/nisq/teleport"
    file_type = "pkl"

class NISQTelegateDataMgr(BaseDataMgr):
    data_dir = "data/nisq/telegate"
    file_type = "pkl"