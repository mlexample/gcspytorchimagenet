"""
gcsdataset provides a torchvision-like ImageFolder dataset that works on
regular files as well as google cloud storage. It also supports caching for
expensive recursive metdata operations.
"""
import io
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.cloud.storage.blob import Blob
from google.cloud.storage.client import Client
from PIL import Image

from torchvision.datasets.vision import VisionDataset

import torchvision
import torchvision.transforms as transforms
import torch_xla.utils.gcsfs

_CLIENT = None

class SizedDataLoader(object):
    def __init__(self, loader, size):
        self.loader = loader
        self.size = size
    def __iter__(self):
        return iter(self.loader)
    def __len__(self):
        return self.size

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Client()
    return _CLIENT


def make_dataset(
        paths: List[str],
        extensions: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[Tuple[str, int]], Dict[bytes, int]]:
    """
    Map folder+classnames into list of (imagepath, class_index). Requires potentially
    expensive glob filesystem. For large object store datasets it's recommended to cache this.
    """
    instances = []
    counter = 0
    if extensions is None:
        extensions = ()
    # make it easier to directly match on result of str.split
    ext_set = set(ext[1:].lower() for ext in extensions)
    classes_to_idx = {}
    for path in sorted(paths):
        components = path.split('/')
        # based on above glob expression the last 2 components are filename/class
        potentialclass = components[-2].encode('utf-8')
        fname = components[-1]
        if fname.split('.')[-1].lower() not in ext_set:
            continue
        if potentialclass not in classes_to_idx:
            classes_to_idx[potentialclass] = counter
            counter += 1
        instances.append((path, classes_to_idx[potentialclass]))
    return instances, classes_to_idx


def _write_bytes(path: str, data: str):
    with open(path, 'w') as handle:
        handle.write(data)


def _read_bytes(path: str):
    try:
        with open(path, 'rb') as handle:
            return handle.read()
    except FileNotFoundError:
        raise NotFoundException()


def _gcs_write_bytes(path: str, data: str):
    # skip caching if local
    blob = Blob.from_string(path)
    # There seems to be a bug in upload_from_string not using client properly
    blob.bucket._client = _get_client()  # pylint: disable=protected-access
    blob.upload_from_string(data)


def _gcs_read_bytes(path: str):
    try:
        buf = io.BytesIO()
        _get_client().download_blob_to_file(path, buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        raise NotFoundException()


class NotFoundException(Exception):
    """
    Thrown when a filesystem or gcs resource isn't found.
    """


class ImageFolder(VisionDataset):
    """
    ImageFolder is a work-alike of the torchvision ImageFolder class supporting gcs.
    """

    def __init__(
            self,
            root: str,
            index_path: Optional[str] = None,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(ImageFolder, self).__init__(root, transform=transform,
                                          target_transform=target_transform)

        load_from_cache = True
        samples = None
        paths = None
        self.write_cb = _write_bytes
        self.read_cb = _read_bytes
        if index_path is not None and "gs://" in index_path:
            self.write_cb = _gcs_write_bytes
            self.read_cb = _gcs_read_bytes
        if index_path is not None:
            try:
                paths = json.loads(self.read_cb(index_path))
            except NotFoundException:
                pass
        if paths is None:
            load_from_cache = False
            # note: relying on private api to avoid some extra stat calls.
            #pylint: disable-protected-access
            ls = torch_xla._XLAC._xla_tffs_list # pytype: disable=module-attr
            #pylint: enable-protected-access
            paths = list(ls(os.path.join(self.root, "*", "*.JPEG")))
            paths.sort()
        samples, classes_to_idx = make_dataset(
            paths,
            torchvision.datasets.folder.IMG_EXTENSIONS)  # pytype: disable=module-attr
        # pylint: disable=len-as-condition
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
        """
        Loader will attempt to read contents of uri (only regular file paths and gs:// urls
        are supported. It returns a PIL Image object.
        """
        buf = self._buf
        buf.seek(0)
        buf.write(self.read_cb(uri))
        img = Image.open(buf).convert('RGB')
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


def _main():
    import sys
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_dim = 224
    if 'IMAGE_DIR' not in os.environ:
        raise Exception("IMAGE_DIR env variable is required")
    directory = os.environ['IMAGE_DIR']
    train_dataset = ImageFolder(
        root=directory+"/train",
        index_path=directory+'/imagenetindex_train.json',
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
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0


if __name__ == "__main__":
    _main()
