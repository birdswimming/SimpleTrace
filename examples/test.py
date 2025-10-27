import json
import os
import pickle
import random
import time

from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

import simpletrace

simpletrace.add_trace_dataset_function("pipeline.transforms")

arr = [[random.random() for _ in range(100)] for _ in range(100)]
with open("/tmp/random_matrix.pkl", "wb") as f:
    pickle.dump(arr, f)

json_data = {
    "name": "random_data",
    "count": 10,
    "items": [
        {
            "id": i,
            "value": random.randint(1, 100),
            "tag": random.choice(["A", "B", "C", "D"]),
        }
        for i in range(10)
    ],
}

# 保存为 json 文件
with open("/tmp/test.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)


class Callable_Test:
    def __init__(self):
        pass

    def __call__(self, idx):
        time.sleep(0.1)
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本文件夹路径

        with open(os.path.join(base_dir, "/tmp/random_matrix.pkl"), "rb") as f:
            loaded_matrix = pickle.load(f)

        with open(os.path.join(base_dir, "/tmp/test.json"), "rb") as f:
            loaded_json = json.load(f)

        return idx


class Compose:
    def __init__(self, len):
        self.transforms = [Callable_Test() for _ in range(len)]

    def __call__(self, idx):
        for t in self.transforms:
            idx = t(idx)
        return idx


class SimpleDataset(Dataset):
    def __init__(self, len):
        super().__init__()
        self.pipeline = Compose(2)

    def __len__(self):
        return 1000  # 假设数据量是100

    # def pipeline(self, idx):
    #     time.sleep(0.2)
    #     return idx

    def __getitem__(self, idx):
        time.sleep(0.1)
        result = self.pipeline(idx)
        time.sleep(0.1)
        return result


# dataset = SimpleDataset(2)

# # 顺序采样器
# sampler = SequentialSampler(dataset)

# # 批采样器：按batch size分组，不打乱顺序
# batch_sampler = BatchSampler(sampler, batch_size=5, drop_last=False)

# # 创建 DataLoader：使用多个 worker，默认 collate_fn
# dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory= True)

dataset = SimpleDataset(2)
# 顺序采样器
sampler = SequentialSampler(dataset)
# 批采样器：按batch size分组，不打乱顺序
batch_sampler = BatchSampler(sampler, batch_size=5, drop_last=False)
# 创建 DataLoader：使用多个 worker，默认 collate_fn
dataloader = DataLoader(
    dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=True
)

for _ in dataloader:
    time.sleep(1)

time.sleep(10)
