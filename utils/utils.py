import random
import numpy as np
import json
import torch
import os

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


def sample_image_labels(
    labels,
    num_images,
    seed,
    label_source,
    manifest_path,
):
    if num_images <= 0:
        raise ValueError("num_images must be positive")

    label_source = os.path.normpath(str(label_source))
    if os.path.exists(manifest_path):
        with open(manifest_path) as file:
            manifest = json.load(file)
        expected = {
            "seed": int(seed),
            "num_images": int(num_images),
            "label_source": label_source,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(
                    f"sampling manifest conflict for {field}: "
                    f"expected {value!r}, found {manifest.get(field)!r}"
                )
        images = manifest.get("images", [])
        expected_count = min(int(num_images), len(labels))
        if manifest.get("sampled_count") != expected_count:
            raise ValueError(
                "sampling manifest count does not match the current label dict"
            )
        missing = [path for path in images if path not in labels]
        if missing:
            raise ValueError(
                f"sampling manifest contains missing image: {missing[0]}"
            )
    else:
        candidates = sorted(labels)
        sample_size = min(int(num_images), len(candidates))
        images = random.Random(int(seed)).sample(candidates, sample_size)
        manifest = {
            "seed": int(seed),
            "num_images": int(num_images),
            "sampled_count": sample_size,
            "label_source": label_source,
            "images": images,
        }
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w") as file:
            json.dump(manifest, file, indent=4)

    return {path: labels[path] for path in images}
