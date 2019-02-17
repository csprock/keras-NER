from keras.layers import Dense, Embedding, LSTM, Dropout, TimeDistributed, Add, Conv1D, MaxPooling1D, Input, Flatten, GlobalMaxPooling1D
from keras.layers import Bidirectional, concatenate
from keras.models import Model
from keras.initializers import RandomUniform
from keras_contrib.layers import CRF

def blstm_cnn_wd_ft_ner(max_len_sent, 
                        max_len_word, 
                        num_tags, 
                        word_embedding_dims, 
                        word_feature_embedding_dims, 
                        char_embedding_dims,
                        recurrent_dropout=0,
                        char_dropout=0.5,
                        main_dropout=0.68,
                        filters=53,
                        fc_dropout=0,
                        kernel_size = 3):
    '''
    BLSTM-CNN network with word features. 
    '''
    ##### character-level CNN ######
    char_input = Input(shape = (None, max_len_word,), name='char_input')
    char_embedding = TimeDistributed(Embedding(input_dim=char_embedding_dims[0],
                                               output_dim=char_embedding_dims[1],
                                               input_length=max_len_word,
                                               #mask_zero = True,
                                               embeddings_initializer=RandomUniform(minval = -0.5, maxval = 0.5),
                                               name = 'char_embedding'))(char_input)
    
    
    
    X_char = TimeDistributed(Conv1D(filters=filters,
                                    kernel_size=3,
                                    activation='relu',
                                    padding='same',
                                    name='char_layer_1'))(char_embedding)
    
    
    #X_char = TimeDistributed(MaxPooling1D(5, name = 'char_pool_layer'))(X_char)
    X_char = TimeDistributed(GlobalMaxPooling1D(name='char_pool_layer'))(X_char)
    X_char = TimeDistributed(Flatten())(X_char)
    X_char = Dropout(char_dropout)(X_char)
    
    ############# word-level embeddings ##############
    ##### word-level feature embedding ######
    word_feature_input = Input(shape=(None,), name='word_feature_input')
    word_feature_embedding = Embedding(input_dim=word_feature_embedding_dims[0],
                                       output_dim=word_feature_embedding_dims[1],
                                       input_length=None,
                                       mask_zero=True)(word_feature_input)
    
    ###### word embedding #####
    word_input = Input(shape=(None,), name='word_input')

    word_embedding = Embedding(input_dim=word_embedding_dims[0],
                               output_dim=word_embedding_dims[1],
                               input_length=None,
                               mask_zero=True,
                               name='word_embedding')(word_input)

    ##### main BLSTM network #####
    X = concatenate([X_char, word_embedding, word_feature_embedding])
    
    X = Bidirectional(LSTM(units=max_len_sent,
                           recurrent_dropout=recurrent_dropout,
                           dropout=main_dropout,
                           return_sequences=True,
                           name='main_layer_1'),
                           merge_mode='concat')(X) # TODO: change to concat

    #X = TimeDistributed(Dense(100, activation='relu'))(X)
    #X = TimeDistributed(Dropout(fc_dropout))(X)
    X = TimeDistributed(Dense(num_tags, activation='tanh'))(X)
    #X = TimeDistributed(Dropout(fc_dropout))(X)

    crf = CRF(num_tags)
    out = crf(X)

    model = Model(inputs = [word_input, word_feature_input, char_input], outputs = out)
    return model
    
