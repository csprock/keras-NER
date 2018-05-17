import json
import itertools
import numpy as np
from feature_utils import featureTransformer

##################################
########## load data #############
##################################

# load CoNLL2003
train = json.load(open('C:/Users/csprock/Documents/Projects/NER/data/conll2003/en/train.json'))
valid = json.load(open('C:/Users/csprock/Documents/Projects/NER/data/conll2003/en/valid.json'))
test = json.load(open('C:/Users/csprock/Documents/Projects/NER/data/conll2003/en/test.json'))


##########################################
#### initialize feature transformer ######
##########################################

###### initialize vocabulary and tags #######

WORDS, TAGS = list(), list()
for d in itertools.chain(train['data'], valid['data'], test['data']):
    for w in d['sentence']:
        if w.lower() not in WORDS: WORDS.append(w.lower())
            
    for t in d['tags']:
        if t not in TAGS: TAGS.append(t)
            

MAX_LEN_SENT = 50   # maximum sentence length
MAX_LEN_WORD = 15   # maximum word length

# initialize feature transformers 
feat_transformer = featureTransformer(WORDS, TAGS, tag_pad_value = 'O', zero_tag = 'O', 
                                      word_padding = 'post', word_truncating = 'post', 
                                      sent_truncating = 'post', sent_padding = 'post', 
                                      max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_WORD)

#################################
#### make training tensors ######
#################################

X_sent_train, X_char_train, X_word_ft_train, Y_train = feat_transformer.makeTensors(train['data'], sentences = True, characters = True, word_features = True, tags = True)
X_sent_val, X_char_val, X_word_ft_val, Y_val = feat_transformer.makeTensors(valid['data'], sentences = True, characters = True, word_features = True, tags = True)
X_sent_test, X_char_test, X_word_ft_test, Y_test = feat_transformer.makeTensors(test['data'], sentences = True, characters = True, word_features = True, tags = True)


###################################
##### prepare word embeddings #####
###################################

emb_dir = 'C:/Users/csprock/Documents/Projects/NER/embeddings/glove.6B/glove.6B.100d.txt'

e = open(emb_dir, encoding = 'UTF-8')

embeddings = dict()
for line in e:
    values = line.split()
    word = values[0]
    coef = np.asarray(values[1:], dtype = 'float32')
    embeddings[word] = coef
    
e.close()

embedding_dim = (len(feat_transformer.word2idx), 100)
E = np.zeros(embedding_dim)

for i, w in enumerate(feat_transformer.word2idx):
    emb_vec = embeddings.get(w)
    if emb_vec is not None:
        E[i,:] = emb_vec



###################################
########## model training #########
###################################
from sklearn.metrics import classification_report

from keras.callbacks import EarlyStopping
from models.blstm_cnn_word_features_model import blstm_cnn_wd_ft_ner
from keras.optimizers import SGD

model = blstm_cnn_wd_ft_ner(max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_WORD, num_tags = len(feat_transformer.tag2idx), 
                            word_embedding_dims = embedding_dim, 
                            char_embedding_dims = (len(feat_transformer.char2idx), 25),
                            word_feature_embedding_dims = (6,4))
                            


model.layers[7].set_weights([E])
model.layers[7].trainable = False

model.compile(optimizer = SGD(lr = 0.01), loss = 'categorical_crossentropy', metrics = ['acc'])

early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.0001, patience = 9)


model.fit(x = {'word_input':X_sent_train, 'word_feature_input':X_word_ft_train, 'char_input':X_char_train}, 
          y = Y_train, 
          batch_size = 32, 
          validation_data = ({'word_input':X_sent_val, 'word_feature_input':X_word_ft_val, 'char_input':X_char_val}, Y_val), 
          epochs = 75, 
          callbacks = [early_stopping])



####### prediction #######
Y_pred = model.predict(x = {'word_input':X_sent_test, 'word_feature_input':X_word_ft_test, 'char_input':X_char_test})

# unroll predicted tensors
Y_pred = np.argmax(Y_pred, axis = 2).reshape((Y_pred.shape[0]*Y_pred.shape[1],1))
Y_test = np.argmax(Y_test, axis = 2).reshape((Y_pred.shape[0]*Y_pred.shape[1],1))

# convert indices back to tags
y_pred, y_true = [None]*Y_pred.shape[0], [None]*Y_test.shape[0]
for i in range(Y_pred.shape[0]):
    y_pred[i] = feat_transformer.idx2tag[Y_pred[i][0]]
    y_true[i] = feat_transformer.idx2tag[Y_test[i][0]]


print(classification_report(y_true = y_true, y_pred = y_pred))












