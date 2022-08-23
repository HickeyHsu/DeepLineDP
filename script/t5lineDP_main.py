import os, re, argparse
import random
import sys
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

import torch.optim as optim

import numpy as np
import pandas as pd
from transformers import AdamW, AutoConfig, AutoTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

from tqdm import tqdm

from sklearn.utils import compute_class_weight

from CoLineDP_model import *
from my_util import *
from preprocess_data import MyDataset
from _logger import logger
from preprocess_data import create_code_df
class TextDataset(Dataset):
    """ 读取数据文件，对函数源代码进行编码，生成编码后的数据集 """
    def __init__(self, tokenizer, args, data:pd.DataFrame,max_train_LOC=900,max_sent_len=None):
        
        #读取对应的数据文件
        self.examples=[]
        srcs = data["SRC"].tolist()# SRC (str): The original file written in C/C++
        labels = data["target"].tolist() # target (int): The file-level label that determines whether a function is vulnerable or not
        RelFilename = data["RelFilename"].tolist()        
        for i in tqdm(range(len(srcs))):
            #首先分割成行
            code_df=create_code_df(srcs[i],RelFilename[i])
            code_df_line=code_df[((code_df['is_comment']==False)&(code_df['is_blank']==False))]
            code_lines=code_df_line['code_line'].tolist()
            file_input=convert_files_to_features(code_lines, labels[i], tokenizer, args)
            self.examples.append(file_input)#源代码to ids
        max_lines=max([len(xs.input_tokens) for xs in self.examples])
        logger.info(f"*** max_lines = {max_lines} ***")
        if max_sent_len is None:
            self.max_sent_len = min(max([len(sent.input_ids) for sent in self.examples]), max_train_LOC)
        else:
            self.max_sent_len=max_sent_len
        #补足长度
        pad_line_ids=[0]*self.max_sent_len
        for file in self.examples:
            if len(file.input_ids) < self.max_sent_len:
                need_lens=self.max_sent_len-len(file.input_ids)
                file.input_ids.append([pad_line_ids]*need_lens)
            else:
                file.input_ids=file.input_ids[:self.max_sent_len]
        max_lens=max([max([len(x) for x in xs.input_tokens]) for xs in self.examples])
        
        # for example in self.examples[:3]:
        #         logger.info("*** Example ***")
        #         logger.info("label: {}".format(example.label))
        #         logger.info("input_tokens: {}".format([[x.replace('\u0120','_') for x in xs] for xs in example.input_tokens]))
        #         logger.info("input_ids: {}".format('\n'.join([' '.join(map(str, x)) for x in example.input_ids])))
        logger.info(f"*** max_lens of line tokens = {max_lens} ***")
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids=self.examples[i].input_ids
        return torch.tensor(input_ids),torch.tensor(self.examples[i].label)
def gen_dataloader(tokenizer, args, data:pd.DataFrame,shuffle=True,max_train_LOC=900,max_sent_len=None):
     #读取对应的数据文件
    examples=[]
    srcs = data["SRC"].tolist()# SRC (str): The original file written in C/C++
    labels = data["target"].tolist() # target (int): The file-level label that determines whether a function is vulnerable or not
    RelFilename = data["RelFilename"].tolist()        
    for i in tqdm(range(len(srcs))):
        #首先分割成行
        code_df=create_code_df(srcs[i],RelFilename[i])
        code_df_line=code_df[((code_df['is_comment']==False)&(code_df['is_blank']==False))]
        code_lines=code_df_line['code_line'].tolist()
        file_input=convert_files_to_features(code_lines, labels[i], tokenizer, args)
        examples.append(file_input)#源代码to ids
    max_lines=max([len(xs.input_tokens) for xs in examples])
    max_ids=max([max([len(x) for x in xs.input_ids]) for xs in examples])
    logger.info(f"*** max_lines = {max_lines} ***")
    logger.info(f"*** max_ids = {max_ids} ***")
    # sys.exit()
    if max_sent_len is None:
        max_sent_len = min(max([len(sent.input_ids) for sent in examples]), max_train_LOC)
    x_vecs=[]#shape:(file_nums,max_sent_len,block_size)
    #补足长度
    pad_line_ids=[tokenizer.pad_token_id]*args.block_size
    logger.info(f"*** len of pad_line_ids = {len(pad_line_ids)} ***")
    for file in examples:
        if len(file.input_ids) < max_sent_len:
            need_lens=max_sent_len-len(file.input_ids)
            for i in range(need_lens):
                file.input_ids.append(pad_line_ids)            
        file.input_ids=file.input_ids[:max_sent_len]
        # x_vecs.append([file.input_ids])
    x_vecs=[file.input_ids for file in examples]
    max_ids_list=[max([len(sentx) for sentx in filex]) for filex in x_vecs]
    # logger.info(f"*** max_ids_list = {max_ids_list} ***")
    max_ids=max(max_ids_list)
    logger.info(f"*** max_ids = {max_ids} ***")
    logger.info(f"{len(x_vecs)},{len(labels)}")
    logger.info(f"*** max_LOC  = {max_sent_len} ***")
    # for example in self.examples[:3]:
    #         logger.info("*** Example ***")
    #         logger.info("label: {}".format(example.label))
    #         logger.info("input_tokens: {}".format([[x.replace('\u0120','_') for x in xs] for xs in example.input_tokens]))
    #         logger.info("input_ids: {}".format('\n'.join([' '.join(map(str, x)) for x in example.input_ids])))
    max_lens=max([max([len(x) for x in xs.input_tokens]) for xs in examples])
    logger.info(f"*** max_lens of line tokens = {max_lens} ***")
    y_tensor =  torch.FloatTensor(labels)
    x_tensor=torch.tensor(x_vecs)
    logger.info(f"x_tensor shape ={x_tensor.shape}")
    tensor_dataset = TensorDataset(x_tensor, y_tensor)
    dl = DataLoader(tensor_dataset,shuffle=shuffle,batch_size=args.train_batch_size,drop_last=True)    
    return dl,max_sent_len

def get_loss_weight(labels,weight_dict):
    '''
        input
            labels: a PyTorch tensor that contains labels
        output
            weight_tensor: a PyTorch tensor that contains weight of defect/clean class
    '''
    label_list = labels.cpu().numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])

    weight_tensor = torch.tensor(weight_list).reshape(-1,1).cuda()
    return weight_tensor

def train_model(args, train_dataset,t5model, tokenizer, eval_dataset):
    model=CoHierarchicalAttentionNetwork(t5model,tokenizer,args)
    model.to(args.device)
    criterion = nn.BCELoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    weight_dict = {}
    train_label=train_dataset["target"].tolist()
    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_label), y = train_label)

    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)
    # build dataloader 数据加载
    train_dataloader,max_sent_len=gen_dataloader(tokenizer,args,train_dataset)
    vali_dataloader,max_sent_len=gen_dataloader(tokenizer,args,train_dataset,max_sent_len=max_sent_len)
    logger.info("train data loaded")
    args.max_steps = args.epochs * len(train_dataloader)#最长训练步数=epoch数*数据批数
    logger.info(f"max_steps ={args.max_steps}")

    # evaluate the model per epoch
    args.save_steps = len(train_dataloader) #验证性能，保存最佳性能下的网络参数
    logger.info(f"save_steps ={args.save_steps}")
    args.warmup_steps = args.max_steps // 5 #前20%为预热学习，学习率慢慢增加；后80%学习率逐渐衰减
    logger.info(f"warmup_steps ={args.warmup_steps}")
    
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0
    train_loss_all_epochs = []
    val_loss_all_epochs = []
    for idx in range(args.epochs): #对于每个epoch
        train_losses = []
        val_losses = []
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0#初始化训练损失
        for step, batch in enumerate(bar):#对于每个batch
            (inputs_ids, labels) = [x.to(args.device) for x in batch]#读取数据
            model.train()#切换到训练模式：即启用batch normalization和dropout，保证BN层能够用到每一批数据的均值和方差
            final_scrs, word_att_weights, sent_att_weights, sents = model(input_ids=inputs_ids, labels=labels,max_sent_len=max_sent_len)#输入，其中inputs_ids是将token转为词表id后的结果；
            weight_tensor = get_loss_weight(labels,weight_dict)
            criterion.weight = weight_tensor

            loss = criterion(final_scrs, labels.reshape(args.batch_size,1))

            train_losses.append(loss.item())            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()#总训练损失
            tr_num += 1
            train_loss += loss.item()#本epoch训练损失
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,np.mean(train_losses)))
        train_loss_all_epochs.append(np.mean(train_losses))
        optimizer.step()#更新所有参数
        optimizer.zero_grad()#梯度归零
        scheduler.step()
        with torch.no_grad():
            
            criterion.weight = None
            model.eval()
            bar_vali = tqdm(vali_dataloader,total=len(vali_dataloader))
            for step, batch in enumerate(bar):#对于每个batch

                (inputs_ids, labels) = [x.to(args.device) for x in batch]#读取数据
                output, _, __, ___ = model(input_ids=inputs_ids, labels=labels,max_sent_len=max_sent_len)
            
                val_loss = criterion(output, labels.reshape(args.batch_size,1))

                val_losses.append(val_loss.item())

            val_loss_all_epochs.append(np.mean(val_losses))

            logger.info('- at epoch:',str(idx))
            actual_save_model_dir = os.path.join(args.output_dir,args.dataset)
            if not os.path.exists(actual_save_model_dir):
                os.makedirs(actual_save_model_dir)
            if args.exp_name == '':
                torch.save({
                            'epoch': idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+str(idx)+'epochs.pth')
            else:
                torch.save({
                            'epoch': idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+args.exp_name+'_'+str(idx)+'epochs.pth')

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        
        loss_df.to_csv(os.path.join(actual_save_model_dir,args.dataset+'-loss_record.csv'),index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='groovy', help='software project name (lowercase)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=50, help='word embedding size')
    parser.add_argument('--word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')
    parser.add_argument('--sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')
    parser.add_argument('--word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
    parser.add_argument('--sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--exp_name',type=str,default='')
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input training data dir.")
    parser.add_argument("--test_size", default=0.2, type=float, help="test_size.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    
    
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    """ set随机seed 用于复现结果 """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)   

    all_releases = {
        'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
        'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
        'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
        'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
        'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
        'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
        'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
        'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']
    }
    
    my_dataset=MyDataset(all_releases,args)
    train,eval,test=my_dataset.get_file_data_split(args.dataset)
    logger.info(f"my_dataset was loaded")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<str>", "<char>"])
    logger.info(f"tokenizer was loaded from{args.tokenizer_name}")

    t5model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)    
    t5model.resize_token_embeddings(len(tokenizer))
    # logger.info(f"T5ForConditionalGeneration was loaded from{t5model}")
    logger.info(f"T5ForConditionalGeneration was loaded from{args.model_name_or_path}")
    
    # Training
    if args.do_train:
        # train_dataset = TextDataset(tokenizer, args, train)#训练集
        # logger.info("train_dataset loaded")
        # eval_dataset = TextDataset(tokenizer, args, eval)#验证集
        # logger.info("eval_dataset loaded")
        train_model(args, train, t5model, tokenizer, eval)#训练

if __name__ == "__main__":
    main()