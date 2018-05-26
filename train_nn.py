import json
import itertools
import numpy as np
from feature_utils import TensorMaker

##################################
########## load data #############
##################################

def convert_keys(data):
    if isinstance(data, dict):
        if 'data' in data.keys():
            temp = {}
            for k, v in data['data'].items(): temp[int(k)] = v
            data['data'] = temp
            return data
        else:
            return data
    else:
        return data

# load CoNLL2003
train = json.load(open('C:/Users/csprock/Documents/Projects/NER/data/conll2003/en/train.json'), object_hook = convert_keys)
valid = json.load(open('C:/Users/csprock/Documents/Projects/NER/data/conll2003/en/valid.json'), object_hook = convert_keys)
test = json.load(open('C:/Users/csprock/Documents/Projects/NER/data/conll2003/en/test.json'), object_hook = convert_keys)


###### initialize vocabulary and tags #######

WORDS, TAGS = list(), list()
for _, d in itertools.chain(train['data'].items(), valid['data'].items(), test['data'].items()):
    for w in d['sentence']:
        if w.lower() not in WORDS: WORDS.append(w.lower())
            
    for t in d['tags']:
        if t not in TAGS: TAGS.append(t)
            
##### initialize TensorMaker ######

MAX_LEN_SENT = 100   # maximum sentence length
MAX_LEN_WORD = 15    # maximum word length

TM = TensorMaker(WORDS, TAGS, max_len_word = MAX_LEN_WORD, word_padding = 'post', word_truncating = 'post')


###################################
##### prepare word embeddings #####
###################################

emb_dir = 'C:/Users/csprock/Documents/Projects/NER/embeddings/glove.6B/glove.6B.50d.txt'

e = open(emb_dir, encoding = 'UTF-8')

embeddings = dict()
for line in e:
    values = line.split()
    word = values[0]
    coef = np.asarray(values[1:], dtype = 'float32')
    embeddings[word] = coef
    
e.close()

embedding_dim = (len(TM.word2idx), 50)
E = np.zeros(embedding_dim)

for i, w in enumerate(TM.word2idx):
    emb_vec = embeddings.get(w)
    if emb_vec is not None:
        E[i,:] = emb_vec



###################################
########## define model  ##########
###################################

from models.blstm_cnn_word_features_model import blstm_cnn_wd_ft_ner

model = blstm_cnn_wd_ft_ner(max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_WORD, num_tags = len(TM.tag2idx), 
                            word_embedding_dims = embedding_dim, 
                            char_embedding_dims = (len(TM.char2idx), 25),
                            word_feature_embedding_dims = (6,4))
             

model.layers[8].set_weights([E])
model.layers[8].trainable = False


###################################
########## model training  ########
###################################


from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from generators import DataGenerator

BATCH_SIZE = 32

model.compile(optimizer = RMSprop(lr = 0.005), loss = 'categorical_crossentropy', metrics = ['acc'])

early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.0001, patience = 9)


DG = DataGenerator(data = train['data'], batch_size = BATCH_SIZE, tensor_maker = TM, shuffle = True, sentences = True, characters = True, word_features = True, tags = True)
VG = DataGenerator(data = valid['data'], batch_size = BATCH_SIZE, tensor_maker = TM, shuffle = True, sentences = True, characters = True, word_features = True, tags = True)

model.fit_generator(generator = DG, validation_data = VG, validation_steps = len(VG), steps_per_epoch = len(DG), epochs = 75, callbacks = [early_stopping], shuffle = True)


#model.save('models/cnn_blstm_wd_ft_3_d100.h5')
###################################
##########    test  ###############
###################################

from validation import sentence_metrics
from generators import TestDataGenerator

TG = TestDataGenerator(test['data'], BATCH_SIZE, TM, True, True, True, True)

pp, cp, tp = 0, 0, 0
for batch in TG:
    
    X_data, Y_test = batch 
    Y_pred = model.predict_on_batch(X_data)
    Y_pred, Y_test= np.argmax(Y_pred, axis = 2), np.argmax(Y_test, axis = 2)
    
    for i in range(Y_pred.shape[0]):
        
        r_tp, r_cp = sentence_metrics(TM.convert2tags(Y_pred[i,:]), TM.convert2tags(Y_test[i,:]))
        _, r_pp = sentence_metrics(TM.convert2tags(Y_test[i,:]), TM.convert2tags(Y_pred[i,:]))
        cp += r_cp
        tp += r_tp
        pp += r_pp
        
    
recall = (tp/cp)
precision = (tp/pp)

f1 = 2/((1/precision) + (1/recall))
print("Precision: %s, Recall: %s, F1: %s" % (precision, recall, f1))

