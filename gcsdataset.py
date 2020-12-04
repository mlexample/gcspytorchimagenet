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
        classes_to_idx: Dict[bytes, int],
        extensions: Optional[Tuple[str, ...]]=None
) -> List[Tuple[str, int]]:
    """
    Map folder+classnames into list of (imagepath, class_index). Requires potentially expensive glob of
    virtual filesystem. For large object store datasets it's recommended to cache this.
    """
    # note: relying on private api to avoid some extra stat calls.
    paths = torch_xla._XLAC._xla_tffs_list(os.path.join(directory, "*", "*.JPEG")) # pytype: disable=module-attr
    instances = []
    classes = set(classes_to_idx.keys())
    if extensions is None:
        extensions = ()
    # make it easier to directly match on result of str.split
    extSet = set(ext[1:].lower() for ext in extensions)
    for path in paths:
        components = path.split('/')
        # based on above glob expression the last 2 components are filename/class
        potentialclass = components[-2].encode('utf-8')
        fname = components[-1]
        if potentialclass not in classes:
            continue
        if fname.split('.')[-1].lower() not in extSet:
            continue
        instances.append((path, classes_to_idx[potentialclass]))
    return instances

class ImageFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            synset_path: str,
            synset_loader: Optional[Callable[[str], bytes]] = torch_xla.utils.gcsfs.read,
            index_path: Optional[str] = None,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
) -> None:
      super(ImageFolder, self).__init__(root, transform=transform,
                                          target_transform=target_transform)
      classes, class_to_idx = self._find_classes(synset_path)
      load_from_cache = True
      samples = None
      if index_path is not None:
          f = io.BytesIO()
          try:
              get_client().download_blob_to_file(index_path, f)
              f.seek(0)
              samples = json.loads(f.read())
          except Exception as e:
              print(e)
      if samples is None:         
          load_from_cache = False
          samples = make_dataset(self.root, class_to_idx, torchvision.datasets.folder.IMG_EXTENSIONS)  # pytype: disable=module-attr
      if len(samples) == 0:          
        msg = "Found 0 files in subfolders of: {}\n".format(self.root)
        if extensions is not None:
          msg += "Supported extensions are: {}".format(",".join(extensions))
          raise RuntimeError(msg)
        
      self.classes = classes
      self.class_to_idx = class_to_idx
      self.samples = samples
      self.targets = [s[1] for s in samples]
      self.imgs = self.samples
      if index_path is not None and not load_from_cache:
          self._cache_index(index_path)

    def loader(self, uri):
      f = self._buf
      f.seek(0)
      if uri.startswith("gs://"):
          get_client().download_blob_to_file(uri, f)
      else:
          with open(uri, "rb") as handle:
              f.write(handle.read())
      img = Image.open(f).convert('RGB')
      return img
    def _find_classes(self, synset_path: str) -> Tuple[List[bytes], Dict[bytes, int]]:
      # Read categories from file.        
      classes = []
      # Small hack to differentiate between gcs and local paths
      if synset_path.startswith("gs://"):
          data = torch_xla.utils.gcsfs.read(synset_path)
      else:
          with open(synset_path, "rb") as f:
              data=f.read()
      for cls in data.split(b'\n'):
          cls = cls.strip()
          if cls == '':
              continue
          classes.append(cls)
      
      classes.sort()
      class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
      return classes, class_to_idx
    def _cache_index(self, fname: str) -> None:
        # upload_
        blob = Blob.from_string(fname)
        # There seems to be a bug in upload_from_string not using client properly
        blob.bucket._client = get_client()
        blob.upload_from_string(json.dumps(self.samples))
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
    directory=os.environ['IMAGE_DIR']
    train_dataset = ImageFolder(
        root=directory+"/train",
        synset_path=directory+"/synset_labels.txt",
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
        synset_path=directory+"/synset_labels.txt",
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
            if i%100 == 0:
                avg = (i+1)/(time.time()-t0)
                print("Loaded {} samples. Average {} samples/s".format( (i+1), avg))
