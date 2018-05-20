import json
import itertools
import numpy as np
from feature_utils import featureTransformer, TensorMaker
from batch_tools import create_batch_indices

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

#training_batch_indices = create_batch_indices(train['data'], batch_size = 32, separate = False, shuffle = True)
#valid_batch_indices = create_batch_indices(valid['data'], batch_size = 32, separate = False)
#test_batch_indices = create_batch_indices(test['data'], batch_size = 32, separate = False)

##########################################
#### initialize feature transformer ######
##########################################

###### initialize vocabulary and tags #######

WORDS, TAGS = list(), list()
for _, d in itertools.chain(train['data'].items(), valid['data'].items(), test['data'].items()):
    for w in d['sentence']:
        if w.lower() not in WORDS: WORDS.append(w.lower())
            
    for t in d['tags']:
        if t not in TAGS: TAGS.append(t)
            

MAX_LEN_SENT = 100   # maximum sentence length
MAX_LEN_WORD = 15   # maximum word length


TM = TensorMaker(WORDS, TAGS, max_len_word = MAX_LEN_WORD, word_padding = 'post', word_truncating = 'post')



#TM.makeTensors(test['data'][1], 6)
## initialize feature transformers 
#feat_transformer = featureTransformer(WORDS, TAGS, tag_pad_value = 'O', zero_tag = 'O', 
#                                      word_padding = 'post', word_truncating = 'post', 
#                                      sent_truncating = 'post', sent_padding = 'post', 
#                                      max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_WORD)

##################################
##### make training tensors ######
##################################


#X_sent_train, X_char_train, X_word_ft_train, Y_train = feat_transformer.makeTensors(train['data'], sentences = True, characters = True, word_features = True, tags = True)
#X_sent_val, X_char_val, X_word_ft_val, Y_val = feat_transformer.makeTensors(valid['data'], sentences = True, characters = True, word_features = True, tags = True)
#X_sent_test, X_char_test, X_word_ft_test, Y_test = feat_transformer.makeTensors(test['data'], sentences = True, characters = True, word_features = True, tags = True)
#
def test_generator(index_set, data, tm):
    
    def get_sentences(indices, data):
        return [data[i] for i in indices]    

    for _, batch in index_set.items():
        temp_sent = get_sentences(batch, data)
        L = len(temp_sent[0]['sentence'])
        
        X_sent, X_char, X_word_ft, Y = tm.makeTensors(temp_sent, L, sentences = True, 
                             characters = True, 
                             word_features = True, 
                             tags = True)
        
        yield X_sent, X_word_ft, X_char, Y
        




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
########## model training #########
###################################
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from models.blstm_cnn_word_features_model import blstm_cnn_wd_ft_ner2
#from models.blstm_cnn_model import blstm_cnn_ner
#from models.blstm_model import blstm_ner

model = blstm_cnn_wd_ft_ner2(max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_WORD, num_tags = len(TM.tag2idx), 
                            word_embedding_dims = embedding_dim, 
                            char_embedding_dims = (len(TM.char2idx), 25),
                            word_feature_embedding_dims = (6,4))
             

#model2 = blstm_cnn_ner(max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_WORD, num_tags = len(feat_transformer.tag2idx), 
#                            word_embedding_dims = embedding_dim, 
#                            char_embedding_dims = (len(feat_transformer.char2idx), 25))

#model = blstm_ner(MAX_LEN_SENT, embedding_dim, len(feat_transformer.tag2idx))    


model.layers[7].set_weights([E])
model.layers[7].trainable = False

model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['acc'])

early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.0001, patience = 9)

from generators import DataGenerator
DG = DataGenerator(batch_size = 32, tensor_maker = TM, data = train['data'], shuffle = True)
VG = DataGenerator(batch_size = 32, tensor_maker = TM, data = valid['data'], shuffle = True)

model.fit_generator(generator = DG, validation_data = VG, validation_steps = 142, steps_per_epoch = 477, epochs = 20, callbacks = [early_stopping], shuffle = False)

#model.save('models/cnn_blstm_wd_ft_1.h5')

#test_batch_indices = create_batch_indices(test['data'], batch_size = 32, separate = False)
#PG = test_generator(test_batch_indices, test['data'], TM)
#
#Y_pred = list()
#for batch in PG:
#    X_sent, X_word_ft, X_char, _ = batch 
#    Y_temp = model.predict_on_batch({'word_input':X_sent, 'word_feature_input':X_word_ft, 'char_input':X_char})
#    Y_pred.append(Y_temp)
#    
#    
#Y_test = np.stack(tuple(Y_pred), axis = 2)
#
#
#
#
#PG = DataGenerator(test_batch_indices, tm, test['data'])
#Y_pred = model.predict_generator(generator = PG, steps = len(PG))
#
#
#
#
#Y_pred = np.argmax(Y_pred, axis = 2)
#Y_test = np.argmax(Y_test, axis = 2)
#
#
#pp, cp, tp = 0, 0, 0
#for i in range(Y_pred.shape[0]):
#    
#    r_tp, r_cp = sentence_metrics(feat_transformer.convert2tags(Y_pred[i,:]), feat_transformer.convert2tags(Y_test[i,:]))
#    _, r_pp = sentence_metrics(feat_transformer.convert2tags(Y_test[i,:]), feat_transformer.convert2tags(Y_pred[i,:]))
#    cp += r_cp
#    tp += r_tp
#    pp += r_pp
#    
#    

#recall = (tp/cp)
#precision = (tp/pp)
#
#f1 = 2/((1/precision) + (1/recall))
#print(f1)

