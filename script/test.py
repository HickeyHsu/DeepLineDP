import pandas as pd
import os, re, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold # 分层分割
from preprocess_data import create_code_df
char_to_remove = ['+','-','*','/','=','++','--','\\','<str>','<char>','|','&','!']
class MyDataset:
    def __init__(self,
                all_releases:dict,
                data_root_dir = '../datasets/original/',
                # save_dir = "../datasets/preprocessed_data/",
                test_size=0.2
                ):
    
        self.all_releases=all_releases
        self.data_root_dir=data_root_dir        
        self.test_size=test_size

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # self.save_dir=save_dir
        self.file_lvl_dir = os.path.join(data_root_dir,'File-level')
        self.line_lvl_dir = os.path.join(data_root_dir,'Line-level')
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
        # train_eval = all_data.iloc[int(len(all_data)*self.test_size):].reset_index(drop = True)
        # test  = all_data.iloc[:int(len(all_data)*self.test_size)].reset_index(drop = True)    
        # train = train_eval.iloc[int(len(train_eval)*self.test_size):].reset_index(drop = True)
        # eval = train_eval.iloc[:int(len(train_eval)*self.test_size)].reset_index(drop = True)
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

if __name__ == "__main__":
    all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
        'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
        'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
        'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
        'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
        'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
        'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
        'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}
    data_dir=r"D:\data_sci\line-level-defect-prediction\Dataset"
    md=MyDataset(all_releases,data_dir)
    
    train,eval,test=md.get_file_data_split('groovy')
    src=eval.iloc[1]['SRC']
    code_df = create_code_df(src, "test")
    print(code_df)