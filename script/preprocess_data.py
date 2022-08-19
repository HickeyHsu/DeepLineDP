import pandas as pd
import os, re
import numpy as np
from sklearn.model_selection import StratifiedKFold

from my_util import *



char_to_remove = ['+','-','*','/','=','++','--','\\','<str>','<char>','|','&','!']
class MyDataset:
    def __init__(self,all_releases:dict,args):
    
        self.all_releases=all_releases
        self.data_dir=args.data_dir
        self.test_size=args.test_size
        self.file_lvl_dir = os.path.join(self.data_dir,'File-level')
        self.line_lvl_dir = os.path.join(self.data_dir,'Line-level')
    def preprocess_data(self,proj_name):
        cur_all_rel = self.all_releases[proj_name]
        result={}
        for rel in cur_all_rel:
            result[rel]=self.get_rel_data(rel)            
        return result
    def get_cp_file_data_split(self):
        train_list=[]
        eval_list=[]
        test_list=[]        
        for proj_name in self.all_releases.keys():
            train,eval,test=self.get_file_data_split(proj_name)
            train_list.append(train)
            eval_list.append(eval)
            test_list.append(test)
        train_set=pd.concat(train_list,ignore_index=True)
        eval_set=pd.concat(eval_list,ignore_index=True)
        test_set=pd.concat(test_list,ignore_index=True)
        return train_set,eval_set,test_set
    def get_file_data_split(self,proj_name):
        cur_all_rel = self.all_releases[proj_name]
        df_list=[]
        for rel in cur_all_rel:
            df=pd.read_csv(os.path.join(self.file_lvl_dir,rel+'_ground-truth-files_dataset.csv'), encoding='latin')
            df['Project']=proj_name
            df['Release']=rel
            df['RelFilename']=df['Release']+'$'+df['File']
            df_list.append(df)
        all_data=pd.concat(df_list,ignore_index=True)
        all_data['target']=all_data['Bug'].astype(int)
        skf = StratifiedKFold(n_splits=5)
        t=all_data.Bug
        ids=[]
        for train_index, test_index in skf.split(np.zeros(len(t)), t):
            ids.append((train_index, test_index))
        train_index, test_index=ids[0]
        train_eval = all_data.loc[train_index].reset_index(drop=True)
        test = all_data.loc[test_index].reset_index(drop=True)
       
        t=train_eval.Bug
        ids=[]
        for train_index, test_index in skf.split(np.zeros(len(t)), t):
            ids.append((train_index, test_index))
        train_index, test_index=ids[0]
        train = train_eval.loc[train_index].reset_index(drop=True)
        eval = train_eval.loc[test_index].reset_index(drop=True)
        return train,eval,test
    def get_rel_data(self,rel):
        file_level_data = pd.read_csv(os.path.join(self.file_lvl_dir,rel+'_ground-truth-files_dataset.csv'), encoding='latin')
        line_level_data = pd.read_csv(os.path.join(self.line_lvl_dir,rel+'_defective_lines_dataset.csv'), encoding='latin')        
        buggy_files = list(line_level_data['File'].unique())

        preprocessed_df_list = []

        for idx, row in file_level_data.iterrows():
            
            filename = row['File']

            if '.java' not in filename:
                continue

            code = row['SRC']
            label = row['Bug']

            code_df = create_code_df(code, filename)
            code_df['file-label'] = [label]*len(code_df)
            code_df['line-label'] = [False]*len(code_df)

            if filename in buggy_files:
                buggy_lines = list(line_level_data[line_level_data['File']==filename]['Line_number'])
                code_df['line-label'] = code_df['line_number'].isin(buggy_lines)

            if len(code_df) > 0:
                preprocessed_df_list.append(code_df)

        return pd.concat(preprocessed_df_list)
    def get_dataset(self):
        for proj in list(self.all_releases.keys()):
            self.preprocess_data(proj)



data_root_dir = '../datasets/original/'
save_dir = "../datasets/preprocessed_data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_lvl_dir = data_root_dir+'File-level/'
line_lvl_dir = data_root_dir+'Line-level/'


def is_comment_line(code_line, comments_list):
    '''
        input
            code_line (string): source code in a line
            comments_list (list): a list that contains every comments
        output
            boolean value
    '''

    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith('//'):
        return True
    elif code_line in comments_list:
        return True
    
    return False

def is_empty_line(code_line):
    '''
        input
            code_line (string)
        output
            boolean value
    '''

    if len(code_line.strip()) == 0:
        return True

    return False

def preprocess_code_line(code_line):
    '''
        input
            code_line (string)
    '''

    code_line = re.sub("\'\'", "\'", code_line)
    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)
    code_line = re.sub('\b\d+\b','',code_line)
    code_line = re.sub("\\[.*?\\]", '', code_line)
    code_line = re.sub("[\\.|,|:|;|{|}|(|)]", ' ', code_line)

    for char in char_to_remove:
        code_line = code_line.replace(char,' ')

    code_line = code_line.strip()

    return code_line

def create_code_df(code_str, filename):
    '''
        input
            code_str (string): a source code
            filename (string): a file name of source code

        output
            code_df (DataFrame): a dataframe of source code that contains the following columns
            - code_line (str): source code in a line
            - line_number (str): line number of source code line
            - is_comment (bool): boolean which indicates if a line is comment
            - is_blank_line(bool): boolean which indicates if a line is blank
    '''

    df = pd.DataFrame()

    code_lines = code_str.splitlines()
    
    preprocess_code_lines = []
    is_comments = []
    is_blank_line = []


    comments = re.findall(r'(/\*[\s\S]*?\*/)',code_str,re.DOTALL)
    comments_str = '\n'.join(comments)
    comments_list = comments_str.split('\n')

    for l in code_lines:
        l = l.strip()
        is_comment = is_comment_line(l,comments_list)
        is_comments.append(is_comment)
        # preprocess code here then check empty line...

        if not is_comment:
            l = preprocess_code_line(l)
            
        is_blank_line.append(is_empty_line(l))
        preprocess_code_lines.append(l)

    if 'test' in filename:
        is_test = True
    else:
        is_test = False

    df['filename'] = [filename]*len(code_lines)
    df['is_test_file'] = [is_test]*len(code_lines)
    df['code_line'] = preprocess_code_lines
    df['line_number'] = np.arange(1,len(code_lines)+1)
    df['is_comment'] = is_comments
    df['is_blank'] = is_blank_line

    return df

def preprocess_data(proj_name):

    cur_all_rel = all_releases[proj_name]

    for rel in cur_all_rel:
        file_level_data = pd.read_csv(file_lvl_dir+rel+'_ground-truth-files_dataset.csv', encoding='latin')
        line_level_data = pd.read_csv(line_lvl_dir+rel+'_defective_lines_dataset.csv', encoding='latin')

        file_level_data = file_level_data.fillna('')

        buggy_files = list(line_level_data['File'].unique())

        preprocessed_df_list = []

        for idx, row in file_level_data.iterrows():

            filename = row['File']

            if '.java' not in filename:
                continue

            code = row['SRC']
            label = row['Bug']

            code_df = create_code_df(code, filename)
            code_df['file-label'] = [label]*len(code_df)
            code_df['line-label'] = [False]*len(code_df)

            if filename in buggy_files:
                buggy_lines = list(line_level_data[line_level_data['File']==filename]['Line_number'])
                code_df['line-label'] = code_df['line_number'].isin(buggy_lines)

            if len(code_df) > 0:
                preprocessed_df_list.append(code_df)

        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(save_dir+rel+".csv",index=False)
        print('finish release {}'.format(rel))
if __name__ == "__main__":

    for proj in list(all_releases.keys()):
        preprocess_data(proj)

