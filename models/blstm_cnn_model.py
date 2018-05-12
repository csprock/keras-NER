from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Conv1D, MaxPooling1D, Input, Flatten
from keras.layers import Bidirectional, concatenate
from keras.models import Model


def blstm_cnn_ner(max_len_sent, max_len_word, num_tags, word_embedding_dims, char_embedding_dims):
    
    ##### character-level CNN ######
    char_input = Input(shape = (max_len_sent, max_len_word,), name = 'char_input')
    char_embedding = TimeDistributed(Embedding(input_dim = char_embedding_dims[0], 
                                               output_dim = char_embedding_dims[1], 
                                               input_length = max_len_word, 
                                               mask_zero = True, 
                                               name = 'char_embedding'))(char_input)
    
    
    
    X_char = TimeDistributed(Conv1D(filters = 30, 
                                    kernel_size = 3, 
                                    activation = 'relu', 
                                    name = 'char_layer_1'))(char_embedding)
    
    
    X_char = TimeDistributed(MaxPooling1D(5, name = 'char_pool_layer'))(X_char)
    X_char = TimeDistributed(Flatten())(X_char)
    
    ##### word-level embedding ######
    word_input = Input(shape = (max_len_sent,), name = 'word_input')

    word_embedding = Embedding(input_dim = word_embedding_dims[0], 
                               output_dim = word_embedding_dims[1], 
                               input_length = max_len_sent, 
                               mask_zero = True, 
                               name = 'word_embedding')(word_input)

    ##### main BLSTM network #####
    X = concatenate([X_char, word_embedding])
    
    X = Bidirectional(LSTM(units = max_len_sent, recurrent_dropout = 0.68, return_sequences = True, name = 'main_layer_1'))(X)
    X = TimeDistributed(Dense(num_tags, activation = 'softmax'))(X)
    
    model = Model(inputs = [word_input, char_input], outputs = X)
    return model
    
    