## 1.Introduction
This is a multimodal dataset that combines audio and visual information.
## 2. How to prepare the dataset
1. Download [raw avmnist dataset](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing)(https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing)
2. Unzip the raw avmnist dataset
3. Modify the path in ./Multibench/multi_model_end2end_test.py: Locate the word `PATH_TO_AVMNIST` and change it to the path to `avmnist` folder, like `/home/xucheng/xh/data/Multimedia/avmnist`
## 3. How to run the code
```bash
   python multi_model_end2end_test.py
```
