import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)
# tensor([[[1.],
#          [1.],
#          [1.],
#          [1.],
#          [0.]]])



#                       enc_input, #enc_input
def get_attn_key_pad_mask(seq_k,     seq_q):
# to mask the padding size of the sequence k
    len_q = seq_q.size(1) # asuming the length of len_q is 5
    padding_mask = seq_k.eq(0) # .eq() means replace the values of 0 with 1
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1) # expand it to (-1, 5, -1)
    return padding_mask
# Output:
# tensor([[[0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 1]]], dtype=torch.uint8)




def get_subsequnet_mask(seq):
    shape = [seq.size(1), seq.size(1)]
    subsequnet_mask = np.triu(np.ones(shape), k=1)
    subsequnet_mask = torch.from_numpy(subsequnet_mask).byte()
    return subsequnet_mask





def get_sinuosoid_encoding_table(position, d_hid, padding_idx = None):

    def cal_angle(posistion, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)





class ScaledDotProductAttention(nn.Module):

    # where temperature  = 8.0, just some reandom name given. It is declared in the multihead attn class where the sqrt of the features is 8
    def __init__(self , attn_dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim = 2)  # sums up all the value from attn to 1, where the softmax ranges from 0 to 1.

    def forward(slef, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(1, 2))
        attn = attn / 8.0

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)  # sums up all the value from attn to 1, where the softmax ranges from 0 to 1.
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn




class MultiHeadAttention(nn.Module):


    def __init__(self, n_heads, d_models, dk, dv, dropout=0.1):
        super.__init__()

        self.n_heads = n_heads    # 8
        self.d_models = d_models  # 512
        self.dk = dk      # vector value
        self.dv = dv

        # TO multiply the q, k, v with the founded weight matrix we do the following

        self.w_qs = nn.linear(d_models, n_heads * dk)
        self.w_ks = nn.linear(d_models, n_heads * dk)
        self.w_vs = nn.linear(d_models, n_heads * dk)

        nn.init.normal_(self.w_qs.weights, mean = 0, std = np.sqrt(2.0/(d_models + dk)))
        nn.init.normal_(self.w_ks.weights, mean = 0, std = np.sqrt(2.0/(d_models + dk)))
        nn.init.normal_(self.w_vs.weights, mean = 0, std = np.sqrt(2.0/(d_models + dv)))

        self.attention = ScaledDotProductAttention()
        self.layerNorm = nn.LayerNorm(d_models)

                           # 64 * 8 = 512, 512
        self.fc = nn.Linear(dk * n_heads, d_models)
        nn.init.xavier_normal_(self.fc.weights)

        self.dropout = nn.Dropout(dropout)

    def forward(slef, q, k, v, maks = None):

        dk, dv, n_heads = self.dk, self.dv, self.n_heads

        # sz_b  = batch size or size of the batch
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # the multiplication actually happens here(q, k, v)
        # q_s: [batch_size x n_heads x len_q x dk]

        q = self.w_qs(q).view(sz_b, len_q, n_heads, dk)
        k = self.w_ks(k).view(sz_b, len_k, n_heads, dk)
        v = self.w_ks(v).view(sz_b, len_v, n_heads, dv)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, dk)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, dk)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, dv)

# working example of .repeat()
# >>> x = torch.tensor([1, 2, 3])
# >>> x.repeat(4, 2)
# tensor([[ 1,  2,  3,  1,  2,  3],
#         [ 1,  2,  3,  1,  2,  3],
#         [ 1,  2,  3,  1,  2,  3],
#         [ 1,  2,  3,  1,  2,  3]])

        mask = mask.repeat(n_heads, 1, 1)
        output, attn = self.attention(q, k, v, mask = mask)

        output = output.view(n_head, sz_b, len_q, dv)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.LayerNorm(output + residual)
# Normalizatoin means multipliying and all the values and bringing #them down to a similar value
# example: the normalized value of the dataset is equal to 1
# layer_norm is used to multiply the values across the matrix(downword)
        return output, attn





class PositionwiseFeedForward(nn.Module):
    # contains two feed forward NN layers

    def __init__(self, din, dout, dropout=0.1):
        super.__init__()

        self.fc1 = nn.Conv1d(din, dout, 1)
        self.fc2 = nn.Conv1d(dout, din, 1)
        self.layer_norm = nn.LayerNorm(din)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.fc2(F.relu(self.fc1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output



# Encoder layer contains
# the output passes through : miltihead -> pos_ffn


class EncoderLayer(nn.Module):

    def __init__(self, d_models, n_heads, dk, dv, din, dropout=0.1):
        super.__init__(EncoderLayer, self)

        self.self_attn = MultiHeadAttention(d_models, n_heads, dk, dv, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_models ,din, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None ,self_attn_mask=None):

        enc_output, enc_self_attn = self.self_attn(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_self_attn



class DecoderLayer(nn.Module):

    def __init__(self, d_models, n_heads, dk, dv, din, dropout = 0.1 ):
        super(DecoderLayer, self)

        self.self_attn = MultiHeadAttention(d_models, n_heads, dk, dv, dropout=dropout)
        self.enc_attn = MultiHeadAttention(d_models, n_heads, dk, dv, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_models ,din, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, self_attn_mask=None, dec_self_attn_mask=None):

        dec_output, dec_self_attn = self.self_attn(dec_input, dec_input, dec_input, mask=self_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_self_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    # d_model = 512, n_heads = 8, n_layer = 6, dk=dv=6
    # n_layer is the number of encoding and decoding layer
    def __init__(self, n_src_vocab, len_max_seq, dim_word_vec,
                n_layers, n_heads, dk, dv, d_models, din,
                dropout=0.1):

        super.__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, dim_word_vec,padding_idx = 0)

        self.position_enc = nn.Embedding.from_pretrained(get_sinuosoid_encoding_table(n_position, dim_word_vec, padding_idx=0), freeze=True)
# the ModuleList is used to perform the operation of the entire class for a given for loop
# In this case it is about creating six EncoderLayers through a for loop, where the n_layers=6
# the ModuleList holds a class inside a list.
        self.layer_stack = nn.ModuleList([EncoderLayer(d_models, din, n_heads, dk, dv, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_self_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)

        non_pad_mask = get_non_pad_mask(src_seq)

        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            if return_attns:
                enc_self_attn += [enc_self_attn]

        if return_attns:
            return enc_output, enc_self_attn_list
        return enc_output,



class Decoder(nn.Module):

    def __init__(self, n_tgt_vocab ,len_max_seq, dim_word_vec, n_layers, n_heads, dk, dv, d_models, din, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_vocab_emb = nn.Embedding(n_tgt_vocab, dim_word_vec, padding_idx = 0)

        self.position_enc = nn.Embedding.from_pretrained(get_sinuosoid_encoding_table(n_position, dim_word_vec, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([DecoderLayer(d_models, din, n_heads, dk, dv, dropout=dropout) for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):


    def __init__(
        self,
        n_src_vocab, n_tgt_vocab, len_max_seq,
        dim_word_vec=512, d_models=512, din=2048,
        n_layers=6, n_heads=8, dk=64, dv=64, dropout=0.1,
        tgt_emb_prj_weight_sharing=True,
        emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            dim_word_vec=dim_word_vec, d_models=d_models, din=din,
            n_layers=n_layers, n_heads=n_heads, dk=dk, dv=dv,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            dim_word_vec=dim_word_vec, d_models=d_models, din=din,
            n_layers=n_layers, n_heads=n_heads, dk=dk, dv=dv,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_models == dim_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_models ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


model = Transformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# the working code is similar to this when given a dataset, prepare it and split them into batches

for epoch in range(20):
    optimizer.zero_grad()
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    loss = criterion(outputs, target_batch.contiguous().view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()


predict, _, _, _ = model(enc_inputs, dec_inputs)




























