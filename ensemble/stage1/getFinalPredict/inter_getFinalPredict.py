import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import os
def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--mixformer_J_Score', 
        type = str)
    parser.add_argument(
        '--mixformer_B_Score', 
        type = str)
    parser.add_argument(
        '--mixformer_JM_Score', 
        type = str)
    parser.add_argument(
        '--mixformer_BM_Score', 
        type = str)
    parser.add_argument(
        '--mixformer_k2_Score', 
        type = str),
    parser.add_argument(
        '--mixformer_k2M_Score', 
        type = str),
    parser.add_argument(
        '--tegcn_J_Score', 
        type = str),
    parser.add_argument(
        '--tegcn_B_Score', 
        type = str),
    parser.add_argument(
        '--ctrgcn_J2d_Score', 
        type = str)
    parser.add_argument(
        '--ctrgcn_B2d_Score', 
        type = str)
    parser.add_argument(
        '--ctrgcn_JM2d_Score', 
        type = str)
    parser.add_argument(
        '--ctrgcn_BM2d_Score', 
        type = str)
    parser.add_argument(
        '--ctrgcn_J3d_Score', 
        type = str),
    parser.add_argument(
        '--ctrgcn_B3d_Score', 
        type = str),
    parser.add_argument(
        '--ctrgcn_JM3d_Score', 
        type = str),
    parser.add_argument(
        '--ctrgcn_BM3d_Score', 
        type = str),
    parser.add_argument(
        '--tdgcn_J2d_Score', 
        type = str)
    parser.add_argument(
        '--tdgcn_B2d_Score', 
        type = str)
    parser.add_argument(
        '--tdgcn_JM2d_Score', 
        type = str)
    parser.add_argument(
        '--tdgcn_BM2d_Score', 
        type = str)
    parser.add_argument(
        '--mstgcn_J2d_Score', 
        type = str)
    parser.add_argument(
        '--mstgcn_B2d_Score', 
        type = str)
    parser.add_argument(
        '--mstgcn_JM2d_Score', 
        type = str)
    parser.add_argument(
        '--mstgcn_BM2d_Score', 
        type = str)
    return parser

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
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = np.load(val_txt_path)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Mix_GCN Score File
    j_file = args.mixformer_J_Score
    b_file = args.mixformer_B_Score
    jm_file = args.mixformer_JM_Score
    bm_file = args.mixformer_BM_Score
    te_file = args.tegcn_J_Score
    teb_file = args.tegcn_B_Score
    k2_file = args.mixformer_k2_Score
    k2m_file = args.mixformer_k2M_Score

    ctrgcn_J_file = args.ctrgcn_J2d_Score
    ctrgcn_B_file = args.ctrgcn_B2d_Score

    ctrgcn_JM_file = args.ctrgcn_JM3d_Score
    ctrgcn_BM_file = args.ctrgcn_BM3d_Score
    
    tdgcn_J_file = args.tdgcn_J2d_Score
    tdgcn_B_file = args.tdgcn_B2d_Score
    tdgcn_JM_file = args.tdgcn_JM2d_Score
    tdgcn_BM_file = args.tdgcn_BM2d_Score
    
    mstgcn_J_file = args.mstgcn_J2d_Score
    mstgcn_B_file = args.mstgcn_B2d_Score
    mstgcn_JM_file = args.mstgcn_JM2d_Score
    mstgcn_BM_file = args.mstgcn_BM2d_Score

    File = [j_file, b_file,jm_file, bm_file, k2_file, k2m_file, te_file, teb_file, \
            ctrgcn_J_file, ctrgcn_B_file, ctrgcn_JM_file, ctrgcn_BM_file, \
            tdgcn_J_file, tdgcn_B_file, tdgcn_JM_file, tdgcn_BM_file,\
            mstgcn_J_file, mstgcn_B_file, mstgcn_JM_file, mstgcn_BM_file,
            ]

    Numclass = 155
    Sample_Num = 4307
    
    Rate = [1035862531236.6625, 84219393602.66, 0.0, 0.0, 
            690226575657.3226, 9380578811.602928, 5092929975302.978, 3428914827973.899, 
            1451180025887.5386, 1634677399308.178, 0.0, 0.0,
            1621190476181.186, 319875314068.55896, 0.0, 0.0, 
            420224497306.3636, 586264313164.3572, 0.0, 0.0, ]

    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    output_dir = '/media/sdd/robot/guosai_ensemble/stage1/getFinalPredict'
    final_score_np = final_score.numpy()

    np.save(os.path.join(output_dir, 'pred.npy'), final_score_np)
    print("集成成功！！！")