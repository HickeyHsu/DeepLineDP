import sys
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from transformers import RobertaForSequenceClassification, T5ForConditionalGeneration


# Model structure
class CoHierarchicalAttentionNetwork(nn.Module):
    def __init__(self, t5model, tokenizer, args):
        """
        vocab_size: number of words in the vocabulary of the model
        embed_dim: dimension of word embeddings
        word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        word_gru_num_layers: number of layers in word-level GRU
        sent_gru_num_layers: number of layers in sentence-level GRU
        word_att_dim: dimension of word-level attention layer
        sent_att_dim: dimension of sentence-level attention layer
        use_layer_norm: whether to use layer normalization
        dropout: dropout rate; 0 to not use dropout
        """
        super(CoHierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(t5model, args.sent_gru_hidden_dim,
                args.sent_gru_num_layers, args.sent_gru_hidden_dim, False, args.dropout)
        self.encoder:T5ForConditionalGeneration = t5model#编码器
        self.fc = nn.Linear(2 * args.sent_gru_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        # self.use_layer_nome = use_layer_norm
        # self.dropout = dropout
        
        # self.tokenizer = tokenizer#
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None,max_sent_len=None):
        
        code_lengths = []
        sent_lengths = []

        for file in input_ids:
            code_line = []
            code_lengths.append(len(file))#[900,900,900,900]
            code_line=[self.args.block_size]*max_sent_len#[96,96,...,96,96](900)
            sent_lengths.append(code_line)#[96,96,...,96,96]

        
        code_tensor = code_tensor.type(torch.LongTensor)
        code_lengths = torch.tensor(code_lengths).type(torch.LongTensor).cuda()
        sent_lengths = torch.tensor(sent_lengths).type(torch.LongTensor).cuda()
        
        code_embeds, word_att_weights, sent_att_weights, sents = self.sent_attention(code_tensor, code_lengths, sent_lengths,output_attentions)

        scores = self.fc(code_embeds)
        final_scrs = self.sig(scores)

        return final_scrs, word_att_weights, sent_att_weights, sents

class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, encoder, sent_gru_hidden_dim,
                sent_gru_num_layers, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()
        # Word-level attention module
        self.encoder=encoder
        # Bidirectional sentence-level GRU
        self.gru = nn.GRU(2 * encoder.model_dim, sent_gru_hidden_dim, num_layers=sent_gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)
    def forward(self, code_tensor, code_lengths, sent_lengths,output_attentions):

        # Sort code_tensor by decreasing order in length 不知道必要性在哪？
        code_lengths, code_perm_idx = code_lengths.sort(dim=0, descending=True)
        code_tensor = code_tensor[code_perm_idx]
        sent_lengths = sent_lengths[code_perm_idx]

        # Make a long batch of sentences by removing pad-sentences
        # i.e. `code_tensor` was of size (num_code_tensor, padded_code_lengths, padded_sent_length)
        # -> `packed_sents.data` is now of size (num_sents, padded_sent_length)
        packed_sents = pack_padded_sequence(code_tensor, lengths=code_lengths.tolist(), batch_first=True)
        last_hidden_state_list=[]
        attentions_list=[]
        for file in code_tensor:
            for input_ids in file:
                outputs = self.encoder.encoder(input_ids, attention_mask=input_ids.ne(0), output_attentions=output_attentions)
                attentions=outputs.attentions
                attentions = attentions[0][0]
                attention = None
                # go into the layer
                for i in range(len(attentions)):
                    layer_attention = attentions[i]
                    # summerize the values of each token dot other tokens
                    layer_attention = sum(layer_attention)
                    if attention is None:
                        attention = layer_attention
                    else:
                        attention += layer_attention
                attention = clean_special_token_values(attention, padding=True)
                attentions_list.append(attention)#注意力
                # 上游的最后一层结果输入到下游分类器
                last_hidden_state_list.append(outputs.last_hidden_state)
        # effective batch size at each timestep
        valid_bsz = packed_sents.batch_sizes

        # Make a long batch of sentence lengths by removing pad-sentences
        # i.e. `sent_lengths` was of size (num_code_tensor, padded_code_lengths)
        # -> `packed_sent_lengths.data` is now of size (num_sents)
        # packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=code_lengths.tolist(), batch_first=True)

    
    
        # Word attention module
        # Sentence-level GRU over sentence embeddings
        sents=torch.tensor(last_hidden_state_list)
        packed_sents, _ = self.gru(PackedSequence(sents, valid_bsz))

        if self.use_layer_norm:
            normed_sents = self.layer_norm(packed_sents.data)
        else:
            normed_sents = packed_sents

        # Sentence attention
        att = torch.tanh(self.sent_attention(normed_sents))
        att = self.sentence_context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val)

        # Restore as documents by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as documents by repadding
        code_tensor, _ = pad_packed_sequence(packed_sents, batch_first=True)

        # Compute document vectors
        code_tensor = code_tensor * sent_att_weights.unsqueeze(2)
        code_tensor = code_tensor.sum(dim=1)

        # Restore as documents by repadding
        word_att_weights=torch.tensor(attention)
        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)

        # Restore the original order of documents (undo the first sorting)
        _, code_tensor_unperm_idx = code_perm_idx.sort(dim=0, descending=False)
        code_tensor = code_tensor[code_tensor_unperm_idx]

        word_att_weights = word_att_weights[code_tensor_unperm_idx]
        sent_att_weights = sent_att_weights[code_tensor_unperm_idx]

        return code_tensor, word_att_weights, sent_att_weights, sents

def clean_special_token_values(all_values, padding=False):
    # special token in the beginning of the seq 
    all_values[0] = 0
    if padding:
        # get the last non-zero value which represents the att score for </s> token
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # special token in the end of the seq 
        all_values[-1] = 0
    return all_values
