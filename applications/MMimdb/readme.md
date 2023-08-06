## 1.Introduction
This is a multimodal dataset about movies, which includes various modal information such as movie information, stills, movie dialogues, and emotional analysis.
## 2. How to prepare the dataset
1. Download [raw mmimdb dataset](https://archive.org/download/mmimdb/mmimdb.tar.gz)(https://archive.org/download/mmimdb/mmimdb.tar.gz)
2. unzip the raw mmimdb dataset
3. Create the `list.txt` file: 
   ```bash
   ls ABSOLUTE_PATH_TO/mmimdb/dataset/*.json > list.txt
   ```
   Use the absolute path so that you can always find the raw data.
## 3. How to run the code
```bash
   python3 multi_model_end2end_test.py
```
