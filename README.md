# MMBench: End-to-End Benchmarking Tool for Analyzing the   Hardware-Software Implications of Multi-modal DNNs

## Ⅰ. Introduction & Background

​	Multi-modal DNNs have become increasingly popular across various application domains due to their significant accuracy improvement compared to SOTA uni-modal DNNs.

​																          				***Multimodal DNN***

​               <img src=".\figures\image-20230726131917741.png" alt="image-20230726131917741" width = 24%  /> <img src=".\figures\image-20230726132014952.png" alt="image-20230726132014952" width = 24% /><img src=".\figures\image-20230726132026796.png" alt="image-20230726132026796" width = 24% /><img src=".\figures\image-20230726132039464.png" alt="image-20230726132039464" width = 24% />
​      *Self-driving&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;             Medical  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;                Multimedia    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               Robotic*

​	To understand the implications of multi-modal DNNs on hardware-software co-designs, we have developed MMBench, an end-to-end benchmarking tool designed to evaluate the performance of multi-modal DNNs at both architecture and system levels.

## II. Overview of MMBench

##### Proposed method

MMBench provides profiling tools based on integrated profilers in both CPU and NVIDIA GPU, including PyTorch profiler, Nsight System, and Nsight Compute. These tools enable researchers to comprehensively understand the execution of multi-modal DNNs. See the figure below for how they work together to analyze DNN performance.

<img src=".\figures\image-20230726132234532.png" alt="image-20230726132234532" style="zoom: 25%;" />

##### Unique features

​	In all, MMBench possesses the following unique features closely related with the characteristics of multi-modal DNNs, which distinguishes itself from general-purpose benchmarks in these specific areas:

- Fine-grained Network Characterization
- End-to-End Application
- ExecutionUser-friendly Profiler Integration 

## Ⅲ. Implementation Details

##### Workloads in MMBench

​	MMBench includes 9 different applications from the five most important multi-modal research domains as shown below. It can cover a wide range of the multi-modal DNNs workloads today.



| **Application**      |     **Domain**      | **Size** | **Modalities**                        | **Unimodal models**      | **Fusion models**          | **Task** **type** |
| -------------------- | :-----------------: | -------- | ------------------------------------- | ------------------------ | -------------------------- | ----------------- |
| Avmnist              |     Multimedia      | Small    | Image<br />Audio                      | CNN                      | Concate/Tensor             | Classification    |
| MMimdb               |     Multimedia      | Medium   | Image <br />Text                      | CNN+transformer          | Concate/Tensor             | Classification    |
| CMU-MOSEI            | Affective computing | Large    | Language<br />Vision<br />Audio       | CNN+transformer          | Concate/Tensor/Transformer | Regression        |
| Sarcasm              | Affective computing | Small    | Language<br />Vision<br />Audio       | CNN+transformer          | Concate/Tensor/Transformer | Classification    |
| Medical VQA          |       Medical       | Large    | Image<br />Text                       | CNN+transformer          | Transformer                | Generation        |
| Medical Segmentation |       Medical       | Large    | MRI scans<br /> (T1, T1c, T2, FLAIR)  | CNN+transformer          | Transformer                | Segmentation      |
| MuJoCo Push          |      Robotics       | Medium   | Image, force, proprioception, control | CNN+RNN                  | Concate/Tensor/Transformer | Classification    |
| Vison & Touch        |      Robotics       | Large    | Image, force, proprioception, depth   | CNN+RNN                  | Concate/Tensor             | Classification    |
| TransFuser           |  Automatic driving  | Large    | Image<br />LiDAR                      | ResNet-34<br />ResNet-18 | Transformer                | Classification    |

<img src=".\figures\image-20230726132314122.png" alt="image-20230726132314122" style="zoom:25%;" />

##### Encoders, fusion and head methods

​	From software aspects, the applications we choose apply many kinds of subnets (mainly as encoders) , fusion ways and head methods, which consititue a whole multi-modal DNN.

<img src=".\figures\image-20230726132334520.png" alt="image-20230726132334520" style="zoom:25%;" />





## Ⅳ. Acknowledgement

Some codes and applications were adapted from the [MultiBench](https://github.com/pliang279/MultiBench).


## Ⅴ. Contributors

Correspondence to: 

  - [Cheng Xu](jerryxu@sjtu.edu.cn) (jerryxu@sjtu.edu.cn)
  - [Xuehan Tang](xuehantang00@gmail.com) (xuehantang00@gmail.com)
  - [Jiacheng Liu](liujiacheng@sjtu.edu.cn) (	liujiacheng@sjtu.edu.cn)
  - [Lingyu Sun](sunlingyu@sjtu.edu.cn) (sunlingyu@sjtu.edu.cn)
  - [Tongqiao Xu](tqxu19@fudan.edu.cn) (tqxu19@fudan.edu.cn)
  - [Peng Tang](85704592@qq.com) (85704592@qq.com)
  - [Tianhao Huang ](hth_2003@sjtu.edu.cn)(hth_2003@sjtu.edu.cn)
  - [Xiaozhi Zhu](zhuxiaozhi@sjtu.edu.cn) (zhuxiaozhi@sjtu.edu.cn)
  - [Mo Niu](2929629852@sjtu.edu.cn) (2929629852@sjtu.edu.cn)
  - [Tianyu Zang](mijiurushi@sjtu.edu.cn) (mijiurushi@sjtu.edu.cn)
  - [Xiaofeng Hou](	houxiaofeng@ust.hk) (houxiaofeng@ust.hk)

## Ⅵ. Related Publications

[**Characterizing and Understanding End-to-End
Multi-modal Neural Networks on GPUs**](https://ieeexplore.ieee.org/abstract/document/9924614)<br>
Xiaofeng Hou, Cheng Xu, Jiacheng Liu, Xuehan Tang, Lingyu Sun, Chao Li and Kwang-Ting Cheng<br>*IEEE Computer Architecture Letters (CAL)*

If you find this repository useful, please cite our paper:

```bibtex
@article{hou2022characterizing,
  title={Characterizing and Understanding End-to-End Multi-modal Neural Networks on GPUs},
  author={Xiaofeng Hou and Cheng Xu and Jiacheng Liu and Xuehan Tang and Lingyu Sun and Chao Li and Kwang-Ting Cheng},
  journal={IEEE Computer Architecture Letters (CAL)},
  year={2022}
}
```xxxxxxxxxx @article{hou2022characterizing,  title={Characterizing and Understanding End-to-End Multi-modal Neural Networks on GPUs},  author={Xiaofeng Hou and Cheng Xu and Jiacheng Liu and Xuehan Tang and Lingyu Sun and Chao Li and Kwang-Ting Cheng},  journal={IEEE Computer Architecture Letters (CAL)},  year={2022}}bibtex
