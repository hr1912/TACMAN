import time
from io import StringIO
from pathlib import Path
import os

import pandas as pd
import psutil

class ResourceRecord:
    def __init__(self):
        self.data = dict()

    def clear(self):
        self.data.clear()

    def get_menory_info(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return dict(
            rss=memory_info.rss,  # 常驻内存  to MB /1024/1024
            vms=memory_info.vms,  # 虚拟内存  to MB /1024/1024
        )

    def record(self, tag: str, is_start: bool = False):
        data = self.get_menory_info()
        data.update(time=time.time())
        if is_start:
            self.data[tag] = dict(start=data)

        else:
            self.data[tag].update(stop=data)

    def to_df(self):
        df = pd.read_csv(StringIO(self.__repr__()), header=None)
        df.columns = ["tag", "type", "time", "rss", "vms"]
        return df

    def save(self, p: Path):
        df = self.to_df()
        df.to_csv(p, index=False)

    def __repr__(self):
        if len(self.data) == 0:
            return ""
        res = []
        for tag, v in self.data.items():
            for k, d in v.items():
                res.append(
                    "{},{},{},{},{}".format(
                        tag, k, str(d["time"]), str(d["rss"]), str(d["vms"])
                    )
                )
        return "\n".join(res)
