import os, re, argparse
import random
import sys
from sklearn import datasets

import torch.optim as optim

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer

from tqdm import tqdm

from sklearn.utils import compute_class_weight

from CoLineDP_model import *
from my_util import *
from preprocess_data import MyDataset
from _logger import logger
class TextDataset(datasets):
    """ 读取数据文件，对函数源代码进行编码，生成编码后的数据集 """
    def __init__(self, tokenizer, args,data:pd.DataFrame):
        #读取对应的数据文件
        self.examples=[]
        srcs = data["SRC"].tolist()# processed_func (str): The original function written in C/C++
        labels = data["target"].tolist() # target (int): The function-level label that determines whether a function is vulnerable or not
        for i in tqdm(range(len(srcs))):
            self.examples.append(convert_examples_to_features(srcs[i], labels[i], tokenizer, args))#源代码to ids
        for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


# def get_loss_weight(labels):
#     '''
#         input
#             labels: a PyTorch tensor that contains labels
#         output
#             weight_tensor: a PyTorch tensor that contains weight of defect/clean class
#     '''
#     label_list = labels.cpu().numpy().squeeze().tolist()
#     weight_list = []

#     for lab in label_list:
#         if lab == 0:
#             weight_list.append(weight_dict['clean'])
#         else:
#             weight_list.append(weight_dict['defect'])

#     weight_tensor = torch.tensor(weight_list).reshape(-1,1).cuda()
#     return weight_tensor

# def train_model(my_dataset:MyDataset,args):
#     weight_dict = {}
#     loss_dir = '../output/loss/DeepLineDP/'
#     actual_save_model_dir = args.save_model_dir+args.dataset+'/'

#     if not args.exp_name == '':
#         actual_save_model_dir = actual_save_model_dir+args.exp_name+'/'
#         loss_dir = loss_dir + args.exp_name

#     if not os.path.exists(actual_save_model_dir):
#         os.makedirs(actual_save_model_dir)

#     if not os.path.exists(loss_dir):
#         os.makedirs(loss_dir)

    
    
    


#     sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train['target']), y = train['target'])

#     weight_dict['defect'] = np.max(sample_weights)
#     weight_dict['clean'] = np.min(sample_weights)
    


#     config = T5Config.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
#     tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
#     t5model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True) 
#     t5model.resize_token_embeddings(len(tokenizer))   

#     train_code3d, train_label = sort_code_by_file(train)
#     valid_code3d, valid_label = sort_code_by_file(eval)
#     x_train_vec = get_x_vec_sa(train_code3d, tokenizer,args.block_size)
#     x_valid_vec = get_x_vec_sa(valid_code3d, tokenizer,args.block_size)
    
#     max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)

#     train_dl = get_dataloader(x_train_vec,train_label,batch_size,max_sent_len)

#     valid_dl = get_dataloader(x_valid_vec, valid_label,batch_size,max_sent_len)

#     model = CoHierarchicalAttentionNetwork(t5model,
#         vocab_size=config.vocab_size,
#         embed_dim=embed_dim,
#         word_gru_hidden_dim=word_gru_hidden_dim,
#         sent_gru_hidden_dim=sent_gru_hidden_dim,
#         word_gru_num_layers=word_gru_num_layers,
#         sent_gru_num_layers=sent_gru_num_layers,
#         word_att_dim=word_att_dim,
#         sent_att_dim=sent_att_dim,
#         use_layer_norm=use_layer_norm,
#         dropout=dropout)

#     model = model.cuda()
#     model.sent_attention.word_attention.freeze_embeddings(False)

#     optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

#     criterion = nn.BCELoss()

#     checkpoint_files = os.listdir(actual_save_model_dir)

#     if '.ipynb_checkpoints' in checkpoint_files:
#         checkpoint_files.remove('.ipynb_checkpoints')

#     total_checkpoints = len(checkpoint_files)

#     # no model is trained 
#     if total_checkpoints == 0:
#         # model.sent_attention.word_attention.init_embeddings(word2vec_weights)
#         current_checkpoint_num = 1

#         train_loss_all_epochs = []
#         val_loss_all_epochs = []
    
#     else:
#         checkpoint_nums = [int(re.findall('\d+',s)[0]) for s in checkpoint_files]
#         current_checkpoint_num = max(checkpoint_nums)

#         checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+str(current_checkpoint_num)+'epochs.pth')
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
#         loss_df = pd.read_csv(loss_dir+dataset_name+'-loss_record.csv')
#         train_loss_all_epochs = list(loss_df['train_loss'])
#         val_loss_all_epochs = list(loss_df['valid_loss'])

#         current_checkpoint_num = current_checkpoint_num+1 # go to next epoch
#         print('continue training model from epoch',current_checkpoint_num)

#     for epoch in tqdm(range(current_checkpoint_num,num_epochs+1)):
#         train_losses = []
#         val_losses = []

#         model.train()

#         for inputs, labels in train_dl:

#             inputs_cuda, labels_cuda = inputs.cuda(), labels.cuda()
#             output, _, __, ___ = model(inputs_cuda)

#             weight_tensor = get_loss_weight(labels)

#             criterion.weight = weight_tensor

#             loss = criterion(output, labels_cuda.reshape(batch_size,1))

#             train_losses.append(loss.item())
            
#             torch.cuda.empty_cache()

#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
#             optimizer.step()

#             torch.cuda.empty_cache()

#         train_loss_all_epochs.append(np.mean(train_losses))

#         with torch.no_grad():
            
#             criterion.weight = None
#             model.eval()
            
#             for inputs, labels in valid_dl:

#                 inputs, labels = inputs.cuda(), labels.cuda()
#                 output, _, __, ___ = model(inputs)
            
#                 val_loss = criterion(output, labels.reshape(batch_size,1))

#                 val_losses.append(val_loss.item())

#             val_loss_all_epochs.append(np.mean(val_losses))

#         if epoch % save_every_epochs == 0:
#             print(dataset_name,'- at epoch:',str(epoch))

#             if exp_name == '':
#                 torch.save({
#                             'epoch': epoch,
#                             'model_state_dict': model.state_dict(),
#                             'optimizer_state_dict': optimizer.state_dict()
#                             }, 
#                             actual_save_model_dir+'checkpoint_'+str(epoch)+'epochs.pth')
#             else:
#                 torch.save({
#                             'epoch': epoch,
#                             'model_state_dict': model.state_dict(),
#                             'optimizer_state_dict': optimizer.state_dict()
#                             }, 
#                             actual_save_model_dir+'checkpoint_'+exp_name+'_'+str(epoch)+'epochs.pth')

#         loss_df = pd.DataFrame()
#         loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
#         loss_df['train_loss'] = train_loss_all_epochs
#         loss_df['valid_loss'] = val_loss_all_epochs
        
#         loss_df.to_csv(loss_dir+dataset_name+'-loss_record.csv',index=False)



def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--dataset',type=str, default='activemq', help='software project name (lowercase)')
    arg.add_argument('--batch_size', type=int, default=32)
    arg.add_argument('--num_epochs', type=int, default=10)
    arg.add_argument('--embed_dim', type=int, default=50, help='word embedding size')
    arg.add_argument('--word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')
    arg.add_argument('--sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')
    arg.add_argument('--word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
    arg.add_argument('--sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
    arg.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    arg.add_argument('--lr', type=float, default=0.001, help='learning rate')
    arg.add_argument('--exp_name',type=str,default='')
    arg.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    arg.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    arg.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input training data dir.")
    arg.add_argument("--test_size", default=0.2, type=float, help="test_size.")
    arg.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    arg.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    arg.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    arg.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    arg.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    arg.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    args = arg.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    """ set随机seed 用于复现结果 """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)   

    all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
        'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
        'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
        'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
        'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
        'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
        'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
        'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}    
    my_dataset=MyDataset(all_releases,args)
    train,eval,test=my_dataset.get_file_data_split(args.dataset)
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    logger.info(f"tokenizer was loaded from{args.tokenizer_name}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)    
    # model.resize_token_embeddings(len(tokenizer))
    logger.info(f"T5ForConditionalGeneration was loaded from{model}")


    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, train)#训练集
        logger.info("train_dataset loaded")
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')#验证集
        logger.info("eval_dataset loaded")
        train(args, train_dataset, model, tokenizer, eval_dataset)#训练

if __name__ == "__main__":
    main()