## I got stuck in the i/o bottleneck in deep learning!

## 深度学习中的CPU、IO瓶颈

# !!! important! I am stilling working on this project, no significant optimization has been done yet!



### 问题的提出

I found large batch size (i.e. 128) can conserve significant amount of time when training a deep learning model (i.e. SRCNN by He. et. al) compared to smaller batch implementation (i.e. 8). Considering the importance of small batch training in fine tuning, I decided to dive to such problem and solve it in a elegant manner.

For the ease of reproducing, I would introduce the original implementation of dataloader.

My training dataset is _DIV2K_ which is available in https://data.vision.ee.ethz.ch/cvl/DIV2K/. Further, I cropped the original image and resized with desired scaling factor. the image patch pairs are stored in folders as `.png` files. Configuration of dataLoader are given as following:

```
self.trainingFeeder = DataLoader(dataset=trainingSet, batch_size=self.batchSize, shuffle=True, num_workers=16, pin_memory=True)
```



在实际训练中，同样多DIV2K图像对进行训练时发现，batchSize为8的时候，20分钟可以训练完整个epoch，而切换到batchSize为128的时候，4分钟就能跑完单个epoch。其中workers数量保持为16，数据是以png格式保存的图片。

#### for ``batch_size=8``, 16 min per epoch:

| GPU  Name        Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| ------------------------------ | -------------------- | -------------------- |
| 0  GeForce GTX 108...  Off     | 00000000:02:00.0  On | N/A                  |

| Fan  Temp  Perf  Pwr:Usage/Cap | Memory-Usage      | GPU-Util  Compute M. |
| ------------------------------ | ----------------- | -------------------- |
| 20%   58C    P2    69W / 250W  | 873MiB / 11170MiB | 33%      Default     |

#### for ``batch_size=16``, 8 min per epoch:

| GPU  Name        Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| ------------------------------ | -------------------- | -------------------- |
| 0  GeForce GTX 108...  Off     | 00000000:02:00.0  On | N/A                  |

| Fan  Temp  Perf  Pwr:Usage/Cap | Memory-Usage      | GPU-Util  Compute M. |
| ------------------------------ | ----------------- | -------------------- |
| 21%   59C    P2   106W / 250W  | 956MiB / 11170MiB | 31%      Default     |

#### for `batch_size=32`, 6 min per epoch:

| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| ----------------------------- | -------------------- | -------------------- |
|   0  GeForce GTX 108...  Off  | 00000000:02:00.0  On |                  N/A |

| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
| ----------------------------- | -------------------- | -------------------- |
|  0%   46C    P2   106W / 250W |    920MiB / 11170MiB |     37%      Default |

#### for `batch_size=64`, 4 min per epoch:
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| ----------------------------- | -------------------- | -------------------- |
|   0  GeForce GTX 108...  Off  | 00000000:02:00.0  On |                  N/A |

| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
| ----------------------------- | -------------------- | -------------------- |
|  0%   53C    P2   154W / 250W |    978MiB / 11170MiB |     52%      Default |

#### for `batch_size=128`, 3 min per epoch:
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| ----------------------------- | -------------------- | -------------------- |
|   0  GeForce GTX 108...  Off  | 00000000:02:00.0  On |                  N/A |

| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
| ----------------------------- | -------------------- | -------------------- |
| 26%   63C    P2   201W / 250W |   1238MiB / 11170MiB |     84%      Default |


### 问题的分析

使用time库查看神经网络子数据准备、推理、更新过程中的时间耗费

![img4](assets/plot4.png '8 workers')

![img3](assets/plot3.png '8 wokers')

![img2](assets/plot2.png '16 workers')

![img1](assets/plot1.png '16 workers')

__Pinned__ memories promise faster copy between CPU memory and GPU memory:

> Host to GPU copies are much faster when they originate from pinned (page-locked) memory. CPU tensors and storages expose a [`pin_memory()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.pin_memory) method, that returns a copy of the object, with data put in a pinned region.

However, at least in this machine, paged memory is much better in smaller batches is slightly better in large batches such as 128.



The growth of data loading&preparing is non-linear, so larger batch can take advantage of that so less percentage of time was taken by data matters.

To improve performance, we should use __less workers__ when dealing with __smaller batches__, __page memory__ if no further optimization is implemented. These tricks are just compromises, __how do we maintain the efficiency in smaller batch execution?__

### 问题的解决

The forwarding and weight updating procedure is too complicated to optimize because well-crafted CUDA code is beyond my fetch. Data loader, which takes non-linear growth when we enlarge the batch, is targeted.

前向传播变化不大，后向传播时间线性增长，这两点耗费是难以进一步优化的。我们在数据加载&处理下功夫。

#### 数据的读取

I currently read `.png` files directly from my SSD, laying heavy burden to CPU to read them into `ndarrays`, this could be a reason that all my CPUs are loaded up to 100% while training (no over loading happens thanks to dynamic loading design of pytorch data loader). Saving the processed data into a binary  file, especially a continuously saved binary file, could be a directly inspired idea.



Note that personally I am by no means a expert in databases, I can't guarantee the depth and robustness of my understanding about __lmdb__, suggestions and corrections are highly appreciated. 



传统的数据读取方法是直接从硬盘中逐个读取

> The reason causing is the slow reading of discountiuous small chunks. 

Months ago I was still using __HDF5__ format as data medians. Though elegant and efficient, h5 files are vulnerable with parallel reading which happens in `torch.utils.DataLoader`(though there is a SWMR feature, I failed to configure the program to a stable state).  _Tensorflow_ has its own `TFRecord` and _MXNET_  do have its `recordIO`, suggested by <https://github.com/Lyken17/Efficient-PyTorch>, _Pytorch_ deserve a better data storage format e.g. __lmdb__ (lightweight Memory-mapped Data Base).

I also found these short-while-clear comparison between __hdf5__ and __lmdb__ in <http://deepdish.io/2015/04/28/creating-lmdb-in-python/>:

>Reasons to use HDF5:
>
>- Simple format to read/write.
>
>Reasons to use LMDB:
>
>- LMDB uses [memory-mapped files](http://en.wikipedia.org/wiki/Memory-mapped_file), giving much better I/O performance.
>- Works well with really large datasets. The HDF5 files are always read entirely into memory, so you can’t have any HDF5 file exceed your memory capacity. You can easily split your data into several HDF5 files though (just put several paths to `h5` files in your text file). Then again, compared to LMDB’s page caching the I/O performance won’t be nearly as good.



##### installation of lmdb

lmdb depends on Cpython so make sure the following libraries are installed:

``` bash
apt-get install libffi-dev python-dev build-essential
```

then just install it

``` bash
pip install lmdb
# or you prefer conda installation
conda install python-lmdb
```

##### structure of lmdb

| Keys                     | Values                   |
| ------------------------ | ------------------------ |
| key01 (type = `bytes()`) | value01 (type=`bytes()`) |

> Always explicitly encode and decode any Unicode values before passing them to LMDB. --- docs

In our implementation, the key would be `bytes(index)`, values would be serialized and compressed by `pyarray` and `lz4framed` as suggested by <https://www.basicml.com/performance/2019/05/18/efficiently-storing-and-retrieving-image-datasets.html>, thanks a lot.

##### structure of my data stored in lmdb

| Keys | Values                                                 |
| ---- | ------------------------------------------------------ |
| b'1' | compress(serialize((`ndarray`, `ndarray`).to_buffer()) |

`to_buffer` method maps the serialized data into continuous memory which can be stored.



[Creating](lmdbPrepare.py) and [reading](lmdbDatabase.py) codes are attached.



One thing that makes __lmdb__ a little annoying is that one should have the map_size bear in mind before writing, thus dealing with image set that doesn't regularize its contained images into a fixed size calls for a good estimation of required size (searched the docs, but no `resize` method was found. Note that hdf5 is resizable).





##### performance

Better, but not so much better :(

![lmdb4](assets/lmdb4.png)

![lmdb3](assets/lmdb3.png)

![lmdb2](assets/lmdb2.png)

![lmdb1](assets/lmdb1.png)

Above results are based on experiments with `input_size = 16`, the __pinned memory__ is not working as told in its documents.  When we choose `input_size = 64` instead, benefit of pinned memory shows up.

![size4](assets/size4.png)

![size3](assets/size3.png)

![size2](assets/size2.png)

![size1](assets/size1.png)
