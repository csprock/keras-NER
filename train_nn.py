import pandas as pd
import numpy as np
from keras.utils import to_categorical
from feature_utils import parse_sentence, sentenceTransformer, tagTransformer


# load data
DATA = pd.read_csv('C:/Users/csprock/Documents/Projects/NER/data/ner_dataset.csv', encoding = 'latin1')
DATA = DATA.fillna(method = 'ffill')
DATA = DATA[['Sentence #', 'Word','Tag']]


TAGS = list(set(DATA['Tag']))
WORDS = list(set(list(map(lambda x: x.lower(), list(set(list(DATA['Word'])))))))


MAX_LEN_SENT = 50   # maximum sentence length
MAX_LEN_WORD = 15   # maximum word length

# initialize feature transformers 
sent_transformer = sentenceTransformer(WORDS, word_padding = 'post', sent_truncating = 'post', word_truncating = 'post', sent_padding = 'post', max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_SENT)
tag_transformer = tagTransformer(TAGS, padding = 'post', truncating = 'post', pad_value = 'O', zero_tag = 'O', max_len_sent = MAX_LEN_SENT)

#################################
#### make training tensors ######
#################################

N = 42000
X_word = [None]*N
X_word_ft = [None]*N

X_char = [None]*N
Y = [None]*N

for i in range(N):
    
    sentence, tags = parse_sentence(i+1, DATA)
    X_word[i] = sent_transformer.wordSequence(sentence)
    X_word_ft[i] = sent_transformer.wordFeatures(sentence)
    
    X_char[i] = sent_transformer.charSequence(sentence)
    Y[i] = tag_transformer.tagSequence(tags)
    

X_sent = sent_transformer.pad_sentences(X_word)
X_word_ft = sent_transformer.pad_sentences(X_word_ft)

X_char = np.dstack(X_char).reshape((N, 50, 15))



Y = tag_transformer.pad_tags(Y)
Y = np.array([to_categorical(i, len(tag_transformer.tag2idx)) for i in Y])

###################################
##### prepare word embeddings #####
###################################

emb_dir = 'embeddings/glove.6B/glove.6B.50d.txt'

e = open(emb_dir, encoding = 'UTF-8')

embeddings = dict()
for line in e:
    values = line.split()
    word = values[0]
    coef = np.asarray(values[1:], dtype = 'float32')
    embeddings[word] = coef
    
e.close()


embedding_dim = (len(sent_transformer.word2idx), MAX_LEN_SENT)
E = np.zeros(embedding_dim)
for i, w in enumerate(sent_transformer.word2idx):
    
    emb_vec = embeddings.get(w)
    if emb_vec is not None:
        E[i,:] = emb_vec



###################################
########## model training #########
###################################

from keras.callbacks import EarlyStopping

model = blstm_cnn_wd_ft_ner(max_len_sent = MAX_LEN_SENT, max_len_word = MAX_LEN_WORD, num_tags = len(tag_transformer.tag2idx), 
                            word_embedding_dims = embedding_dim, 
                            char_embedding_dims = (len(sent_transformer.char2idx), 25),
                            word_feature_embedding_dims = (6,4))
                            

model.layers[7].set_weights([E])
model.layers[7].trainable = False

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', metrics = ['acc'])

early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0.005, patience = 9)

model.fit(x = {'word_input':X_sent, 'word_feature_input':X_word_ft, 'char_input':X_char}, 
          y = Y, batch_size = 64, validation_split = 0.2, epochs = 75, callbacks = [early_stopping])




















