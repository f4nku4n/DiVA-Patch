import random
import numpy as np
import json
import torch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def first_success_query_from_process(
    process, success_index, query_index=None
):
    for query, record in enumerate(process, start=1):
        if bool(record[success_index]):
            if query_index is None:
                return query
            return int(record[query_index])
    return None
