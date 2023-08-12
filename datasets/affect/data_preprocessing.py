import numpy as np
import h5py
import pickle

def extract_subarray(arr):
    # 获取数组的形状
    x, _ = arr.shape
    
    # 检查是否有足够的行数
    if x >= 50:
        # 随机选择50个不全为0的行索引
        row_indices = np.random.choice(np.nonzero((arr != 0).any(axis=1))[0], size=50, replace=False)
        
        # 提取子数组
        subarray = arr[row_indices, :]
        
    else:
        # 计算需要补齐的行数
        rows_to_add = 50 - x
        
        # 随机选择不全为0的行索引
        nonzero_rows = np.nonzero((arr != 0).any(axis=1))[0]
        row_indices = np.random.choice(nonzero_rows, size=rows_to_add, replace=True)
        
        # 补齐原始数组
        subarray = np.concatenate((arr, arr[row_indices]), axis=0)
    
    return subarray

def extract_subarray_text(arr):
    # 获取数组的形状
    x, _ = arr.shape
    
    # 找到非全零行的索引
    nonzero_rows = np.nonzero((arr != 0).any(axis=1))[0]
    if len(nonzero_rows) < 50:
        # 计算需要补齐的行数
        rows_to_add = 50 - len(nonzero_rows)
        
        # 随机选择不全为0的行索引
        additional_rows = np.random.choice(np.setdiff1d(np.arange(x), nonzero_rows), size=max(rows_to_add, 0), replace=True)
        
        # 提取子数组
        subarray = np.concatenate((arr[nonzero_rows], arr[additional_rows]), axis=0)
    else:
        # 随机选择50行非全零的行索引
        selected_rows = np.random.choice(nonzero_rows, size=50, replace=False)
        
        # 提取子数组
        subarray = arr[selected_rows, :]

    # 补齐剩余行数为全零的行
    zero_rows = np.zeros((50 - subarray.shape[0], arr.shape[1]))
    subarray = np.concatenate((subarray, zero_rows), axis=0)
    
    return subarray

def data_preprocess(label_path,audio_path,vision_path,text_path,output_file_path):
    with h5py.File(label_path,'r') as file:
        output_list_label=[]
        output_list_id=[]
        # print(len(list(file['All Labels']['data'].keys())))
        for id in list(file['All Labels']['data'].keys()):
            reshaped_labels_arr=np.expand_dims(file['All Labels']['data'][id]['features'][...][:1,:],axis=0)
            output_list_label.append(reshaped_labels_arr)
            output_list_id.append(id)
    output_label=np.concatenate(output_list_label,axis=0)
    output_id=output_list_id
    # output_label=np.repeat(output_label,10,axis=0)
    # output_id=output_id*10
    print("output_label.shape={}".format(output_label.shape))

    with h5py.File(audio_path, 'r') as file:
        output_list_audio = []
        for id in list(file['COVAREP']['data'].keys()):
            original_arr = file['COVAREP']['data'][id]['features'][...]
            padded_arr = extract_subarray(original_arr)
            # 将填充后的数组添加到列表中
            reshaped_arr = np.expand_dims(padded_arr, axis=0)
            output_list_audio.append(reshaped_arr)
    output_audio = np.concatenate(output_list_audio, axis=0)
    # output_audio=np.repeat(output_audio,10,axis=0)
    print("output_audio.shape={}".format(output_audio.shape))

    with h5py.File(vision_path,'r') as file:
        output_list_vision=[]
        for id in list(file['OpenFace_2']['data'].keys()):
            original_arr=file['OpenFace_2']['data'][id]['features'][...]
            padded_arr=extract_subarray(original_arr)
            reshaped_arr=np.expand_dims(padded_arr,axis=0)
            output_list_vision.append(reshaped_arr)
    output_vision=np.concatenate(output_list_vision,axis=0)
    # output_vision=np.repeat(output_vision,10,axis=0)
    print("output_vision.shape={}".format(output_vision.shape))

    with h5py.File(text_path,'r') as file:
        output_list_text=[]
        for id in list(file['glove_vectors']['data'].keys()):
            original_arr=file['glove_vectors']['data'][id]['features'][...]
            padded_arr=extract_subarray_text(original_arr)
            reshaped_arr=np.expand_dims(padded_arr,axis=0)
            output_list_text.append(reshaped_arr)
    output_text=np.concatenate(output_list_text,axis=0)
    # output_text=np.repeat(output_text,10,axis=0)
    print("output_text.shape={}".format(output_text.shape))
    
    output_dict = {
        'train': {
            'vision': output_vision[:1632],
            'audio': output_audio[:1632],
            'text': output_text[:1632],
            'labels': output_label[:1632],
            'id': output_id[:1632]
        },
        'valid': {
            'vision': output_vision[1632:1632+187],
            'audio': output_audio[1632:1632+187],
            'text': output_text[1632:1632+187],
            'labels': output_label[1632:1632+187],
            'id': output_id[1632:1632+187]
        },
        'test': {
            'vision': output_vision[1632+187:1632+187+466],
            'audio': output_audio[1632+187:1632+187+466],
            'text': output_text[1632+187:1632+187+466],
            'labels': output_label[1632+187:1632+187+466],
            'id': output_id[1632+187:1632+187+466]
        }
    }

    # 使用pickle模块将output_dict保存为.pkl文件
    with open(output_file_path, 'wb') as file:
        pickle.dump(output_dict, file)

def data_preprocess_once(label_path,audio_path,vision_path,text_path,output_file_path):
    with h5py.File(label_path,'r') as file:
        output_list_label=[]
        output_list_id=[]
        for id in list(file['All Labels']['data'].keys())[:3]:
            reshaped_labels_arr=np.expand_dims(file['All Labels']['data'][id]['features'][...][:1,:],axis=0)
            output_list_label.append(reshaped_labels_arr)
            output_list_id.append(id)
    output_label=np.concatenate(output_list_label,axis=0)
    output_id=output_list_id
    print("output_label.shape={}".format(output_label.shape))

    with h5py.File(audio_path, 'r') as file:
        output_list_audio = []
        for id in list(file['COVAREP']['data'].keys())[:3]:
            original_arr = file['COVAREP']['data'][id]['features'][...]
            
            padded_arr = extract_subarray(original_arr)
            
            # 将填充后的数组添加到列表中
            reshaped_arr = np.expand_dims(padded_arr, axis=0)
            output_list_audio.append(reshaped_arr)
    output_audio = np.concatenate(output_list_audio, axis=0)
    print("output_audio.shape={}".format(output_audio.shape))

    with h5py.File(vision_path,'r') as file:
        output_list_vision=[]
        for id in list(file['OpenFace_2']['data'].keys())[:3]:
            original_arr=file['OpenFace_2']['data'][id]['features'][...]
            padded_arr=extract_subarray(original_arr)
            reshaped_arr=np.expand_dims(padded_arr,axis=0)
            output_list_vision.append(reshaped_arr)
    output_vision=np.concatenate(output_list_vision,axis=0)
    print("output_vision.shape={}".format(output_vision.shape))

    with h5py.File(text_path,'r') as file:
        output_list_text=[]
        for id in list(file['glove_vectors']['data'].keys())[:3]:
            original_arr=file['glove_vectors']['data'][id]['features'][...]
            padded_arr=extract_subarray_text(original_arr)
            reshaped_arr=np.expand_dims(padded_arr,axis=0)
            output_list_text.append(reshaped_arr)
    output_text=np.concatenate(output_list_text,axis=0)
    print("output_text.shape={}".format(output_text.shape))

    output_dict = {
        'train': {
            'vision': output_vision[:1],
            'audio': output_audio[:1],
            'text': output_text[:1],
            'labels': output_label[:1],
            'id': output_id[:1]
        },
        'valid': {
            'vision': output_vision[1:2],
            'audio': output_audio[1:2],
            'text': output_text[1:2],
            'labels': output_label[1:2],
            'id': output_id[1:2]
        },
        'test': {
            'vision': output_vision[2:3],
            'audio': output_audio[2:3],
            'text': output_text[2:3],
            'labels': output_label[2:3],
            'id': output_id[2:3]
        }
    }

    # 使用pickle模块将output_dict保存为.pkl文件
    with open(output_file_path, 'wb') as file:
        pickle.dump(output_dict, file)

def data_preprocess_fast(label_path,audio_path,vision_path,text_path,output_file_path):
    with h5py.File(label_path,'r') as file:
        output_list_label=[]
        output_list_id=[]
        # print(len(list(file['All Labels']['data'].keys())))
        for id in list(file['All Labels']['data'].keys())[:50]:
            reshaped_labels_arr=np.expand_dims(file['All Labels']['data'][id]['features'][...][:1,:],axis=0)
            output_list_label.append(reshaped_labels_arr)
            output_list_id.append(id)
    output_label=np.concatenate(output_list_label,axis=0)
    output_id=output_list_id
    # output_label=np.repeat(output_label,10,axis=0)
    # output_id=output_id*10
    # print("output_label.shape={}".format(output_label.shape))

    with h5py.File(audio_path, 'r') as file:
        output_list_audio = []
        for id in list(file['COVAREP']['data'].keys())[:50]:
            original_arr = file['COVAREP']['data'][id]['features'][...]
            padded_arr = extract_subarray(original_arr)
            # 将填充后的数组添加到列表中
            reshaped_arr = np.expand_dims(padded_arr, axis=0)
            output_list_audio.append(reshaped_arr)
    output_audio = np.concatenate(output_list_audio, axis=0)
    # output_audio=np.repeat(output_audio,10,axis=0)
    # print("output_audio.shape={}".format(output_audio.shape))

    with h5py.File(vision_path,'r') as file:
        output_list_vision=[]
        for id in list(file['OpenFace_2']['data'].keys())[:50]:
            original_arr=file['OpenFace_2']['data'][id]['features'][...]
            padded_arr=extract_subarray(original_arr)
            reshaped_arr=np.expand_dims(padded_arr,axis=0)
            output_list_vision.append(reshaped_arr)
    output_vision=np.concatenate(output_list_vision,axis=0)
    # output_vision=np.repeat(output_vision,10,axis=0)
    # print("output_vision.shape={}".format(output_vision.shape))

    with h5py.File(text_path,'r') as file:
        output_list_text=[]
        for id in list(file['glove_vectors']['data'].keys())[:50]:
            original_arr=file['glove_vectors']['data'][id]['features'][...]
            padded_arr=extract_subarray_text(original_arr)
            reshaped_arr=np.expand_dims(padded_arr,axis=0)
            output_list_text.append(reshaped_arr)
    output_text=np.concatenate(output_list_text,axis=0)
    # output_text=np.repeat(output_text,10,axis=0)
    # print("output_text.shape={}".format(output_text.shape))

    output_dict = {
        'train': {
            'vision': output_vision[:1],
            'audio': output_audio[:1],
            'text': output_text[:1],
            'labels': output_label[:1],
            'id': output_id[:1]
        },
        'valid': {
            'vision': output_vision[1:1+40],
            'audio': output_audio[1:1+40],
            'text': output_text[1:1+40],
            'labels': output_label[1:1+40],
            'id': output_id[1:1+40]
        },
        'test': {
            'vision': output_vision[1+40:1+40+2],
            'audio': output_audio[1+40:1+40+2],
            'text': output_text[1+40:1+40+2],
            'labels': output_label[1+40:1+40+2],
            'id': output_id[1+40:1+40+2]
        }
    }

    # 使用pickle模块将output_dict保存为.pkl文件
    # with open(output_file_path, 'wb') as file:
    #     pickle.dump(output_dict, file)
    return output_dict


if __name__ == '__main__':
    data_preprocess_fast(label_path='datasets/affect/raw_data/CMU_MOSEI_Labels.csd', audio_path='datasets/affect/raw_data/CMU_MOSEI_COVAREP.csd', vision_path='datasets/affect/raw_data/CMU_MOSEI_VisualOpenFace2.csd', text_path='datasets/affect/raw_data/CMU_MOSEI_TimestampedWordVectors.csd', output_file_path='datasets/affect/mosei_new.pkl')