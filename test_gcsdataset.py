import os

import unittest
import tempfile
import json

import gcsdataset

# A minimally small valid jpeg file
img = b'\xff\xd8\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10\x10\xff\xc9\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xcc\x00\x06\x00\x10\x10\x05\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9'


class TestGCSDataset(unittest.TestCase):
    def setUp(self):
        self.dirs = ["a", "b"]
        self.directory = tempfile.TemporaryDirectory()
        self.dirname = self.directory.name
        self.count = 0
        self.paths = []
        for key in self.dirs:
            # create the dir/${label}/file.JPEG structure
            path = os.path.join(self.dirname, key)
            os.makedirs(path)
            self.count += 1
            fname = os.path.join(path, "file.JPEG")
            with open(fname, "wb") as f:
                f.write(img)
            self.paths.append(fname)

    def tearDown(self):
        self.directory.cleanup()

    def test_make_dataset(self):
        ds = gcsdataset.make_dataset(
            self.paths, extensions=(".JPEG",))
        self.assertEqual(len(ds), self.count,
                         "Expected {} items got {}".format(self.count, len(ds)))


class TestImageFolder(unittest.TestCase):
    def setUp(self):
        self.classes_to_idx = {b"a": 0, b"b": 1}
        self.directory = tempfile.TemporaryDirectory()
        self.dirname = self.directory.name.encode('utf-8')
        self.count = 0
        self.paths = []
        for key in self.classes_to_idx.keys():
            # create the dir/${label}/file.JPEG structure
            path = os.path.join(self.dirname, key)
            os.makedirs(path)
            self.count += 1
            fname = os.path.join(path, b"file.JPEG")
            with open(fname, "wb") as f:
                f.write(img)
            self.paths.append(fname.decode('utf-8'))
        self.paths.sort()

    def tearDown(self):
        self.directory.cleanup()

    def test_image_folder(self):
        ds = gcsdataset.ImageFolder(root=self.directory.name)
        vals = list(ds)
        self.assertEqual(len(vals), self.count,
                         "Expected {} items got {}".format(self.count, len(vals)))

    def test_index_cache(self):
        index_path = os.path.join(self.directory.name, "vals.json")
        ds = gcsdataset.ImageFolder(
            root=self.directory.name, index_path=index_path)
        self.assertTrue(os.path.exists(index_path), "Index path exists")
        # Bad test practice here: verify cache data is expected structure
        with open(index_path) as f:
            paths = json.loads(f.read())
        self.assertEqual(paths, self.paths)


class TestTrainScript(unittest.TestCase):
    def test_import_train_script(self):
        import test_train_mp_imagenet


if __name__ == '__main__':
    unittest.main()
