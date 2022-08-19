import re
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import T5Tokenizer
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
max_seq_len = 50

all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6', 
                      'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0', 
                      'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'}

all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'], 
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}



all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
                     'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/preprocessed_data/'


word2vec_dir = '../output/Word2Vec_model/' 

def get_df(rel, is_baseline=False):

    if is_baseline:
        df = pd.read_csv('../'+file_lvl_gt+rel+".csv")

    else:
        df = pd.read_csv(file_lvl_gt+rel+".csv")

    df = df.fillna('')

    df = df[df['is_blank']==False]
    df = df[df['is_test_file']==False]

    return df

def prepare_code2d(code_list, to_lowercase = False):
    '''
        input
            code_list (list): list that contains code each line (in str format)
        output
            code2d (nested list): a list that contains list of tokens with padding by '<pad>'
    '''
    code2d = []

    for c in code_list:
        c = re.sub('\\s+',' ',c)

        if to_lowercase:
            c = c.lower()

        token_list = c.strip().split()
        total_tokens = len(token_list)
        
        token_list = token_list[:max_seq_len]

        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>']*(max_seq_len-total_tokens)

        code2d.append(token_list)

    return code2d
    
def get_code3d_and_label(df, to_lowercase = False):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''

    code3d = []
    all_file_label = []

    for filename, group_df in df.groupby('filename'):

        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        code2d = prepare_code2d(code, to_lowercase)
        code3d.append(code2d)

        all_file_label.append(file_label)

    return code3d, all_file_label

def get_w2v_path():

    return word2vec_dir

def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.syn0).cuda()
    
    # add zero vector for unknown tokens
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1,embed_dim).cuda()))

    return word2vec_weights

def pad_code(code_list_3d,max_sent_len,limit_sent_len=True, mode='train'):
    paded = []
    
    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)
            
        
        if mode == 'train':
            if max_sent_len-len(file) > 0:
                for i in range(0,max_sent_len-len(file)):
                    sent_list.append([0]*max_seq_len)

        if limit_sent_len:    
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)
        
    return paded

def get_dataloader(code_vec, label_list,batch_size, max_sent_len):
    y_tensor =  torch.cuda.FloatTensor([label for label in label_list])
    code_vec_pad = pad_code(code_vec,max_sent_len)
    tensor_dataset = TensorDataset(torch.tensor(code_vec_pad), y_tensor)

    dl = DataLoader(tensor_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    
    return dl

def get_x_vec(code_3d, word2vec):
    x_vec = [[[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text]
         for text in texts] for texts in code_3d]
    
    return x_vec

def get_x_vec_sa(code_3d, tokenizer:T5Tokenizer,block_size):
    x_vec = [[convert_examples_to_features(text,tokenizer,block_size)
         for text in texts] for texts in code_3d]
    
    return x_vec

def convert_examples_to_features(func, label, tokenizer, args)->InputFeatures:
    """ 
    源代码encode：将源代码进行分词、映射、对齐，转换为InputFeatures对象保存
    <s>:1,</s>:2,<pad>:3
    """    
    #source：BPE分词
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]#block_size-2:保留<s>和</s>的位置
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]# <s> + code + </s>
    try:
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)# 将tokens转化成单词表中单个字的id，得到id列表
    except:
        print(source_tokens)
        sys.exit()
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length#不同长度编码向量对齐
    return InputFeatures(source_tokens, source_ids, label)#转换为对象保存

def sort_code_by_file(df):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''
    code3d = []
    all_file_label = []
    for filename, group_df in df.groupby('RelFilename'):
        file_label = bool(group_df['Bug'].unique())
        code = list(group_df['SRC'])
        code3d.append(code)
        all_file_label.append(file_label)
    return code3d, all_file_label