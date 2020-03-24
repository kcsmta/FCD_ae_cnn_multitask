from keras.preprocessing import sequence
import numpy as np
# AST data
# each instance (#branches * length of braches)
def load_data2D(path, maxlen=100, padding='post'):
    instances = []
    f = open(path, 'r')
    lines  = f.readlines()
    f.close()
    for idx, instance in enumerate(lines):
        branches = []
        for branch in instance.split('|'):

            branch  = branch.replace('\n','')
            if(branch==''):
                continue
            values = branch.split(',')
            branches.append(values)
        branches =  sequence.pad_sequences(branches, maxlen=maxlen, padding=padding)
        instances.append(branches)
    return np.array(instances)
def load_labels(path):
    labels=[]
    f = open(path, 'r')
    lines  = f.readlines()
    for idx, l in enumerate(lines):
        lines[idx] = lines[idx].replace('\n','')
        labels.append((int)(lines[idx]) - 1)
    f.close()
    return labels
# each instance (#branches * length of braches*1 - channel)
#brach padding: pre,center, post
def load_data3D(path, maxbranch = 300,b_padding ='center',maxlen=100, padding='post', symboldict_path=''):
    instances = []
    f = open(path, 'r')
    lines  = f.readlines()
    symbols_dict ={}
    channel =1
    if symboldict_path !='':
        symbols_dict = loadSymbolDict(symboldict_path)
        channel +=1
    symbols_dict[0] =0

    f.close()
    for idx, instance in enumerate(lines):
        branches = []
        for branch in instance.split('|'):

            branch  = branch.replace('\n','')
            if(branch==''):
                continue
            values = branch.split(',')
            values = pad_seq(values, maxlen=maxlen, padding=padding)
            if channel==1:
                values = [[float(i)] for i in values] # convert to number
            else:
                values = [[float(i), symbols_dict[int(float(i))]] for i in values]  # convert to number
            branches.append(values)
        #branches =  sequence.pad_sequences(branches, maxlen=maxlen, padding=padding)
        instances.append(branches)

        for i in range(0, len(instances)):
            inst_len = len(instances[i])
            if maxbranch < inst_len:# remove branches
                if b_padding=='post':
                    instances[i] = instances[i][0:maxbranch]
                if b_padding=='pre':
                    instances[i] = instances[i][inst_len- maxbranch:inst_len]
                if b_padding=='center':
                    p = (inst_len- maxbranch)/2
                    instances[i] = instances[i][p:maxbranch+p]
            else: # add more branches
                if b_padding=='post':
                    for j in range(0, maxbranch - inst_len):
                        instances[i].append(np.zeros(shape=(maxlen, channel)))
                if b_padding=='pre':
                    for j in range(0, maxbranch - inst_len):
                        instances[i].insert(0,np.zeros(shape=(maxlen, channel)))
                if b_padding=='center':
                    p = (maxbranch - inst_len)/2
                    for j in range(0, p):
                        instances[i].insert(0,np.zeros(shape=(maxlen, channel)))
                    for j in range(p, maxbranch - inst_len):
                        instances[i].append(np.zeros(shape=(maxlen, channel)))

    return np.array(instances)
def pad_seq(values, maxlen=100, padding='post'): # padding: pre or post
    x =[]
    if len(values)>=maxlen:
        if padding == 'post':
            x = values[0: maxlen]
        else:
            x = values[len(values)-maxlen: len(values)]
    else:
        p_len = maxlen - len(values)
        zeros = np.zeros(p_len)
        if padding == 'post':
            x = np.concatenate([values, zeros])
        else:
            x = np.concatenate([zeros, values])

    return np.array(x)
def loadSymbolDict(path='../data/symbols_dict.txt'):
    symbols_dict = {}
    f = open(path, 'r')
    lines = f.readlines()[1:]
    f.close()
    #Symbol,ID,group
    for idx, symbol_info in enumerate(lines):
        values = symbol_info.split(',')
        symbols_dict[int(values[1])] = int(values[2])
    return symbols_dict

# ASM
def loadInstGroup(path='../asmdata/dict_tokType.txt'):
    inst_group = {}
    f = open(path, 'r')
    lines = f.readlines()[1:]
    f.close()
    #Symbol,ID,group
    for line in lines:
        items = line.split()
        inst_group[items[0]] = items[1]
    return inst_group
# read word embedding file: return {(word, vector)}
def loadWordEmbedding(file):
    f = open(file, 'r')
    f.readline() # ignore header
    lines = f.readlines()
    f.close()
    wordvec ={}
    for line in lines:
        items = line.split()
        if len(items)<=1:
            continue
        vec = items[1:]
        vec = [float (i) for i in vec]
        wordvec[items[0]] = vec
    return wordvec

# read a data file
# each line of input data: label sequence
# wordvec: dict of word-vec
#inst_group: dict of instruction-group
# return vector representations of sequences
def load_ASMSeqData(path, wordvec, numview = 1, inst_group = None):
    # labels =[]
    # data = []
    # maxlen =0
    # f = open(path, 'r')
    # lines  = f.readlines()
    # f.close()
    # veclen = len (wordvec.values()[0])
    #
    # for line in lines:
    #     items = line.split()
    #     if len(items)<=1:
    #         continue
    #     labels.append(int (items[0]))
    #
    #     V1 =[]
    #     V2 =[]
    #     for w in items[1:]: # for each word
    #         V1.append(wordvec[w])
    #         if numview == 2:
    #             if w in inst_group:
    #                 group = inst_group[w] # get group of the instruction
    #                 v2 = wordvec[group]
    #             else:
    #                 v2 = np.zeros(shape=veclen).tolist()
    #             V2.append(v2)
    #     if (len(items[1:])) > maxlen:
    #         maxlen = len(items[1:])
    #
    #     if numview==1:
    #         data.append(V1)
    #     else:
    #         data.append([V1, V2])
    labels = []
    dataV1 = []
    dataV2 = []
    maxlen = 0
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    veclen = len(wordvec["movl"])

    for line in lines:
        items = line.split()
        if len(items) <= 1:
            continue
        labels.append(int(items[0]))

        V1 = []
        V2 = []
        for w in items[1:]:  # for each word
            V1.append(wordvec[w])
            if numview == 2:
                if w in inst_group:
                    group = inst_group[w]  # get group of the instruction
                    v2 = wordvec[group]
                else:
                    v2 = np.zeros(shape=veclen).tolist()
                V2.append(v2)
        if (len(items[1:])) > maxlen:
            maxlen = len(items[1:])

        dataV1.append(V1)
        if numview==2:
            dataV2.append(V2)

    if numview==1:
        data = dataV1
    else:
        data =[dataV1, dataV2]
    return labels, data, maxlen

def paddingASMSeq(data, maxword, numview=1): # instance - word - vec (padding word)
    if numview==1:
        len_wordvec =len(data[0][0])
        vec_pad = [0]*len_wordvec
        for inst in data:
            len_padding = maxword - len(inst)
            if len_padding > 0:
                inst.extend (len_padding*[vec_pad]) #(np.zeros(shape=(len_padding, len_wordvec)).tolist())
    else:
        len_wordvec =len(data[0][0][0])
        vec_pad =[0]*len_wordvec # 2 = numview
        for inst in data:
            for v in inst:
                len_padding = maxword - len(v)
                if len_padding>0:
                    v.extend (len_padding* [vec_pad]) #(np.zeros(shape=(len_padding, 2, len_wordvec)).tolist())

def dataStatistics(datafiles,out ):
    fout = open(out, 'w')
    for file in datafiles:
        fin = open(file, 'r')
        lines = fin.readlines()
        fin.close()

        for line in lines:
            items = line.split()
            if len(items) <= 1:
                continue
            # labels.append(int(items[0]))
            fout.write(str(len(items[1:])) +'\n')

    fout.close()

# numview=2
# wordvec = loadWordEmbedding('../asmdata/debug_vec_embedding_no_ops.txt')
# inst_group = loadInstGroup()
# labels, data, maxlen = load_ASMSeqData(path='../asmdata/debug_Seq_CV.txt', wordvec = wordvec, numview = numview, inst_group = inst_group)
# print maxlen,'\n'
# for inst in data:
#     print inst,'\n'
# # paddingASMSeq(data, maxlen, numview=2)
# paddingASMSeq(data[0], maxlen)
# paddingASMSeq(data[1], maxlen)
# print '\nPadding:\n'
# for inst in data:
#     print inst,'\n'
#
# prob ='FLOW016'
# path = '../asmdata/'+ prob
# datafiles =[path+'_Seq_train.txt',path+'_Seq_CV.txt',path+'_Seq_test.txt']
# out = path+'_statistics.csv'
# dataStatistics(datafiles=datafiles, out = out)
#
# prob ='MNMX'
# path = '../asmdata/'+ prob
# datafiles =[path+'_Seq_train.txt',path+'_Seq_CV.txt',path+'_Seq_test.txt']
# out = path+'_statistics.csv'
# dataStatistics(datafiles=datafiles, out = out)
#
# prob ='SUBINC'
# path = '../asmdata/'+ prob
# datafiles =[path+'_Seq_train.txt',path+'_Seq_CV.txt',path+'_Seq_test.txt']
# out = path+'_statistics.csv'
# dataStatistics(datafiles=datafiles, out = out)
#
# prob ='SUMTRIAN'
# path = '../asmdata/'+ prob
# datafiles =[path+'_Seq_train.txt',path+'_Seq_CV.txt',path+'_Seq_test.txt']
# out = path+'_statistics.csv'
# dataStatistics(datafiles=datafiles, out = out)
