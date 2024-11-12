import torch
import pickle
import numpy as np
import pandas as pd
import os

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    # print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    # print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = np.load(val_txt_path)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label


File=[
    # 'wait2ensembleData/orig-val/tdgcn-B-noEnhance.pkl',
    # 'wait2ensembleData/orig-val/tdgcn-B-move.pkl',
    # 'wait2ensembleData/orig-val/tdgcn-B-shift.pkl',
    # 'wait2ensembleData/orig-val/tdgcn-B-rot.pkl',
    # 'wait2ensembleData/orig-val/tdgcn-B.pkl'
]


name = 'tegcn_J.npy'

output_dir = '/media/sdd/robot/guosai_ensemble/stage0'

if __name__ == "__main__":

    Numclass = 155
    Sample_Num = 2000

    Rate=[1/len(File) for _ in File]
    
    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    final_score_np = final_score.numpy()
    print("集成成功！！！")

    array_dict = {f'test_{i}': final_score_np[i] for i in range(final_score_np.shape[0])}

    output_path = os.path.join(output_dir, name.replace('npy','pkl'))

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存字典为 pkl 文件
    with open(output_path, 'wb') as f:
        pickle.dump(array_dict, f)

    print(f"字典已保存到 {output_path}")

    with open(output_path, 'rb') as f:
        data_dict = pickle.load(f)
    nums = data_dict['test_0']
    keys = data_dict.keys()
