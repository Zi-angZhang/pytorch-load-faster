import lmdb

from torch.utils.data import Dataset
from toolkit.PairTransforms import ToTensor, randomRotateFlip, ToFloat
from torchvision.transforms import Compose

from pyarrow import deserialize
from lz4framed import decompress

defaultTransform = Compose([ToFloat(), ToTensor(), randomRotateFlip()])

class lmdbDatabase(Dataset):
    def __init__(self, lmdbFilePath, transform = defaultTransform):
        self.transform = transform

        self.lmdbFile = lmdb.open(lmdbFilePath, subdir=False, readonly=True, lock=False, readahead=False)

        with self.lmdbFile.begin(write=False) as lmdbReader:
            self.length = lmdbReader.stat()['entries']

    def __getitem__(self, index):
        with self.lmdbFile.begin(write=False) as lmdbReader:
            compressedValue = lmdbReader.get(u'{}'.format(index).encode('ascii'))
        valueTuple = deserialize(decompress(compressedValue))
        return self.transform(valueTuple)

    def __len__(self):
        return self.length

