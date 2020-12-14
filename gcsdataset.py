import io
import json
import os
import time
import tempfile

from PIL import Image
from google.cloud.storage.client import Client
from google.cloud.storage.blob import Blob
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torchvision
import torchvision.transforms as transforms
import torch_xla.utils.gcsfs

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Client()
    return _client


def make_dataset(
        directory: str,
        paths: List[str],
        extensions: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[Tuple[str, int]], Dict[bytes, int]]:
    """
    Map folder+classnames into list of (imagepath, class_index). Requires potentially expensive glob of
    virtual filesystem. For large object store datasets it's recommended to cache this.
    """
    instances = []
    counter = 0
    if extensions is None:
        extensions = ()
    # make it easier to directly match on result of str.split
    extSet = set(ext[1:].lower() for ext in extensions)
    classes_to_idx = {}
    for path in sorted(paths):
        components = path.split('/')
        # based on above glob expression the last 2 components are filename/class
        potentialclass = components[-2].encode('utf-8')
        fname = components[-1]
        if fname.split('.')[-1].lower() not in extSet:
            continue
        if potentialclass not in classes_to_idx:
            classes_to_idx[potentialclass] = counter
            counter += 1
        instances.append((path, classes_to_idx[potentialclass]))
    return instances, classes_to_idx


def write_bytes(path: str, data: str):
    with open(path, 'w') as f:
        f.write(data)


def read_bytes(path: str):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        raise NotFoundException()


def gcs_write_bytes(path: str, data: str):
    # skip caching if local
    blob = Blob.from_string(path)
    # There seems to be a bug in upload_from_string not using client properly
    blob.bucket._client = get_client()
    blob.upload_from_string(data)


def gcs_read_bytes(path: str):
    try:
        f = io.BytesIO()
        get_client().download_blob_to_file(path, f)
        f.seek(0)
        return f.read()
    except Exception:  # TODO narrow this
        raise NotFoundException()


class NotFoundException(Exception):
    pass


class ImageFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        index_path: Optional[str] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(ImageFolder, self).__init__(root, transform=transform,
                                          target_transform=target_transform)

        load_from_cache = True
        samples = None
        paths = None
        self.write_cb = write_bytes
        self.read_cb = read_bytes
        if index_path is not None and "gs://" in index_path:
            self.write_cb = gcs_write_bytes
            self.read_cb = gcs_read_bytes
        if index_path is not None:
            try:
                paths = json.loads(self.read_cb(index_path))
            except NotFoundException:
                pass
        if paths is None:
            load_from_cache = False
            # note: relying on private api to avoid some extra stat calls.
            paths = list(torch_xla._XLAC._xla_tffs_list(os.path.join(
                self.root, "*", "*.JPEG")))  # pytype: disable=module-attr
            paths.sort()
        samples, classes_to_idx = make_dataset(
            self.root, paths, torchvision.datasets.folder.IMG_EXTENSIONS)  # pytype: disable=module-attr

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(
                    ",".join(extensions))
                raise RuntimeError(msg)
        classes = list(classes_to_idx.keys())
        classes.sort()
        self.classes = classes
        self.class_to_idx = classes_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
        if index_path is not None and not load_from_cache:
            self._cache_index(index_path, paths)
        self._buf = io.BytesIO()

    def loader(self, uri):
        f = self._buf
        f.seek(0)
        f.write(self.read_cb(uri))
        img = Image.open(f).convert('RGB')
        return img

    def _cache_index(self, fname: str, paths: List[str]) -> None:
        self.write_cb(fname, json.dumps(paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


if __name__ == "__main__":
    import sys
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_dim = 224
    if 'IMAGE_DIR' not in os.environ:
        raise Exception("IMAGE_DIR env variable is required")
    directory = os.environ['IMAGE_DIR']
    train_dataset = ImageFolder(
        root=directory+"/train",
        index_path=directory+'/imagenetindex.json',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    resize_dim = max(img_dim, 256)
    val_dataset = ImageFolder(
        root=directory+"/val",
        index_path=directory+'/imagenetindex_val.json',
        transform=transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_dim),
            transforms.ToTensor(),
            normalize,
        ]))
    if len(sys.argv) == 2 and sys.argv[1] == 'cache':
        # Just ensure data is cached
        sys.exit(0)
    # proceed with regular training
    # Simulate training ops by doing a quick test
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    # benchmark

    for dataset in (train_dataset, val_dataset):
        t0 = time.time()
        print("Begin load of 100 samples")
        print(dataset[0])
        for i in range(0, 100):
            sample = dataset[i]
            if i % 100 == 0:
                avg = (i+1)/(time.time()-t0)
                print("Loaded {} samples. Average {} samples/s".format((i+1), avg))
