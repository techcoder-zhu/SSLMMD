import pandas as pd
import os
import numpy as np

class Pipeline:
    def __init__(self,  ratio, root):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self, src_dir):
        path = src_dir                  #  src_dir="/dataset/"
        for project in os.listdir(path):     #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表 比如ant-1.7.0 camel-1.0.0
            data = None
            for i in os.listdir(path + project):
                if '.csv' in i:
                    data = pd.read_csv(path + project + '/' + i)  #读取那个.csv文件
                    break
            if os.path.exists(self.root + 'versions/' + project + '.pkl') == False:
                data['bug'] = data['bug'].apply(lambda x: 1 if x>0 else x)
                data['id'] = range(data.shape[0])
                data = data.rename(columns = {'name.1':'code', 'bug':'label'})
                data['path']=data['name']+' '+data['version'].map(str)+' '+data['code']

                data_temp = data[['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa','cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']]

                #  对data_temp中的数据进行max-min归一化
                max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
                data_temp = data_temp.apply(max_min_scaler)

                #  对datafrme操作整行整列之后，再进行别的处理就会报错，所以需要copy
                #  https://blog.csdn.net/zaishijizhidian/article/details/98207892        解决地址
                data_temp_copy = data_temp.copy()
                #  对数据整体操作 合并成一个列表
                data_temp_copy.loc[:, 'Traditional'] = data_temp.replace('', np.nan).stack().groupby(level=0).apply(list)

                # 给data新增一列traditional 人工特征
                data_traditional = data_temp_copy['Traditional']
                data = pd.concat([data, data_traditional], axis=1)

                def parse_program(loc):
                    javafile = path + project + '/src/java/' + loc.replace('.','/')+'.java'
                    # add correct path for different projects
##                    if 'xalan' in project:
##                        javafile = javafile + '/src/'
##                    if 'ant' in project:
##                        javafile = javafile + '/src/main/'
##                    if 'poi' in project:
##                        javafile = javafile + '/src/java/'
##                    javafile = javafile + loc.replace('.','/')+'.java'
                    #print (javafile)
                    tree = None
                    if os.path.exists(javafile):
                        fd = open(javafile, "r")
                        try:
                            func = fd.read()
                            import javalang
                            tokens = javalang.tokenizer.tokenize(func)
                            parser = javalang.parser.Parser(tokens)
                            
                            try:
                                tree = parser.parse_compilation_unit()
                            except: # java syntax exception
                                print ('Exception: ' + javafile)
                        except:
                            print ('Reading Exception: ' + javafile)
                    else:
                        print ('Not Found: ' + javafile)
                    return tree
                    
                data['code'] = data['code'].apply(parse_program)
                data = data[data['code'].isnull()==False]
                source = data[['id','code','label','path','Traditional','loc']]
                if not os.path.exists(self.root + 'versions/'):
                    os.mkdir(self.root + 'versions/')
                source.to_pickle(self.root + 'versions/'+ project + '.pkl')

    # merge versions
    def merge_versions(self, project_name):
        if os.path.exists(self.root + project_name + '.pkl') == False:
            path = self.root + 'versions/'
            data = None
            for i in os.listdir(path):
                if project_name in i:
                    if data is None:
                        data = pd.read_pickle(path + i)
                        print (i, len(data))
                    else:
                        data = pd.concat([data, pd.read_pickle(path + i)], ignore_index=True)
                        print (i, len(data))
            data.to_pickle(self.root + project_name + '.pkl')


        
    # split data for training, developing and testing
    def split_data(self, src, dst):
        src_data = pd.read_pickle(self.root + src + '.pkl')
        dst_data = pd.read_pickle(self.root + dst + '.pkl')

        # random under sampling
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=66)
        src_data, y = rus.fit_resample(src_data, src_data['label'])

        # from imblearn.over_sampling import RandomOverSampler as  ros
        # # 对原始数据集进行随机重采样
        # ros = ros(random_state=0)
        # src_data, y = ros.fit_resample(src_data, src_data['label'])


        src_data_num = len(src_data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * src_data_num)
        src_data = src_data.sample(frac=1, random_state=666)
        dst_data = dst_data.sample(frac=1, random_state=666)
        train = src_data.iloc[:train_split]
        dev = src_data.iloc[train_split:]
        test = dst_data

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = self.root + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = self.root + 'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = self.root + 'test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        #str_corpus = [' '.join(c) for c in corpus]
        #trees['code'] = pd.Series(str_corpus)
        #trees.to_csv(self.root+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self,data_path,part):
        from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        #trees = pd.DataFrame(self.sources, copy=True) # from 'clone' dir
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(self.root+part+'/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source('dataset/')
        print('merge versions...')
        project_list = ['xalan', 'ant'] # here can be more projects
        self.merge_versions(project_list[0])
        self.merge_versions(project_list[1]) 
        print('split data...')
        self.split_data('xalan', 'ant') # source project and target project
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline('4:1', 'data/')
ppl.run()


