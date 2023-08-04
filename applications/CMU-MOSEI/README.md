## 如何下载数据集
如果你希望端到端地运行整个网络，那么请参照[CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)，你可以运行那个项目的 `examples/mmdatasdk_examples/full_examples/process_mosei.py` 代码，这是一段处理原始数据的代码，同时可以下载原始数据。

由于这段代码处理数据需要花费大量时间，因此在我们的MMBench中会利用更简单的方式处理数据，仅仅保证数据的格式正确。

## 如何运行这段代码
在下载了数据集之后，将文件CMU_MOSEI_COVAREP.csd,CMU_MOSEI_Labels,CMU_MOSEI_TimestampedWordVectors,CMU_MOSEI_VisualOpenFace2放入到本项目的文件夹 `datasets/affect/raw_data` 下，然后即可运行代码
