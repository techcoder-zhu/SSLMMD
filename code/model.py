import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from transfer_learning import*

class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        if b_in.shape[0] == batch_current.shape[0]:
            self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]



def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    ?????????????????????????????????????????????????????????????????????K
    Params:
	    source: ???????????????n * len(x))
	    target: ??????????????????m * len(y))
	    kernel_mul:
	    kernel_num: ???????????????????????????
	    fix_sigma: ??????????????????sigma???
	Return:
		sum(kernel_val): ?????????????????????
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# ???????????????????????????source???target??????????????????????????????????????????
    total = torch.cat([source, target], dim=0)#???source,target??????????????????
    #???total?????????n+m??????
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #???total???????????????????????????n+m???????????????????????????????????????n+m??????
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #???????????????????????????????????????????????????????????????i,j?????????total??????i???????????????j??????????????????l2 distance(i==j??????0???
    L2_distance = ((total0-total1)**2).sum(2)
    #????????????????????????sigma???
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #???fix_sigma???????????????kernel_mul????????????kernel_num???bandwidth????????????fix_sigma???1????????????[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #?????????????????????????????????
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #????????????????????????
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    ???????????????????????????????????????MMD??????
    Params:
	    source: ???????????????n * len(x))
	    target: ??????????????????m * len(y))
	    kernel_mul:
	    kernel_num: ???????????????????????????
	    fix_sigma: ??????????????????sigma???
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#????????????????????????????????????batchsize??????
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #????????????3?????????????????????4??????
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#??????????????????n==m?????????L???????????????????????????



class BatchProgramClassifier(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramClassifier, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        #class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # self.bigru = nn.LSTM(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
        #                     batch_first=True)
        # linear
        self.hidden2label = nn.Linear((self.hidden_dim * 2)+20, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)
        self.transfer = TCA(kernel_type='primal', dim=220, lamb=1, gamma=1)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def compute_gru_out(self,x,x_feature):
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        # gru  size:(batch_size , length , hidden_dim * 2)
        gru_out, hidden = self.bigru(encodes, self.hidden)
        # print('111',gru_out.size())
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = torch.cat([gru_out, x_feature], dim=1)
        # gru_out = gru_out[:,-1]
        # gru  size:(batch_size , hidden_dim * 2)

        return gru_out

    def forward(self, xs,xs_feature,xs_labels,xt,xt_feature,xt_labels):
        loss = 0
        xs_gru_out = self.compute_gru_out(xs,xs_feature)
        xt_gru_out = self.compute_gru_out(xt,xt_feature)
        # print('222',xt_gru_out.size())
        # XS_gru_out,xs_labels,XT_gru_out,xt_labels = self.transfer.run(xs_gru_out.detach().numpy(),xs_labels.detach().numpy(),xt_gru_out.detach().numpy(),xt_labels.detach().numpy())
        loss = mmd_rbf(xs_gru_out,xt_gru_out)
        y = self.hidden2label(torch.tensor(xs_gru_out,dtype=torch.float32))  # linear
        # y = self.hidden2label(xs_gru_out.clone().datach())

        return y,loss

