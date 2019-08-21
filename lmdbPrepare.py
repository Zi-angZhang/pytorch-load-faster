import numpy
import lmdb
from os import listdir
from os.path import join, exists
import numpy as np
from PIL import Image
from tqdm import tqdm
from pyarrow import serialize
from lz4framed import compress
import random

def keyCart(keyInt):
    return u'{}'.format(keyInt).encode('ascii')

def valueCart(valueInput, valueLabel):
    return compress(serialize((valueInput, valueLabel)).to_buffer())


def lmdbGen(lmdbPath, sourceFolder, crop=False):
    assert not exists(lmdbPath), 'the target file already exists!'
    imageList = [i for i in listdir(sourceFolder) if i.endswith('png')]
    lmdbFile = lmdb.Environment(lmdbPath,
            map_size = int(4e11),
            subdir = False,
            meminit = False,
            map_async = True)
    
    with tqdm(total=len(imageList)) as tbar:
        if crop:
            lmdbWriter = lmdbFile.begin(write=True)
            globalIndex = 0
            for imgName in imageList:
                img = Image.open(join(sourceFolder, imgName))
                h, w = img.size
                for hIndex in range(((h - labelSize) // sampleStep) + 1):
                    for wIndex in range(((w - labelSize) // sampleStep) + 1):
                            hCoor = hIndex * sampleStep
                            wCoor = wIndex * sampleStep

                            hrPatch = img.crop((hCoor, wCoor, hCoor + labelSize, wCoor + labelSize))
                            assert hrPatch.size == (labelSize, labelSize), 'hrPatch size error'
                            lrPatch = hrPatch.resize((inputSize, inputSize), Image.BICUBIC)
                            hrPatch = np.array(hrPatch, dtype=np.int8)
                            lrPatch = np.array(lrPatch, dtype=np.int8)

                            lmdbWriter.put(keyCart(globalIndex), valueCart(lrPatch, hrPatch))
                            globalIndex += 1

                            if globalIndex % 5000 == 0:
                                lmdbWriter.commit()
                                lmdbWriter = lmdbFile.begin(write = True)

                tbar.update()

            lmdbWriter.commit()
            lmdbFile.sync()
            lmdbFile.close()
        
        else:
            print('test dataset mode, no cropping')
            lmdbWriter = lmdbFile.begin(write=True)
            for index, imgName in enumerate(imageList):
                img = Image.open(join(sourceFolder, imgName))
                h, w = img.size
                h = h - (h % scale)
                w = w - (w % scale)
                img = img.crop((0,0,h,w))

                lrImg = img.resize((h//scale, w//scale), Image.BICUBIC)

                lmdbWriter.put(keyCart(index), valueCart(np.array(lrImg)/255, np.array(img)/255))
                
                tbar.update()

            lmdbWriter.commit()
            lmdbFile.sync()
            lmdbFile.close()


                            
if __name__ == '__main__':

    # specially designed for super-resvolution tasks, where inputs and
    # outputs are images. 

    # give input patch size
    inputSize = 16

    # sure images are cropped in a overlapping manner.
    sampleStep = inputSize * 3

    #resize factor
    scale = 4

    labelSize = inputSize * scale

    lmdbGen('ImagePatches/div2k_16_x4.db', '/home/resolution/Dataset/Img/DIV2K/DIV2K_train_HR', crop=True)
