from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

def blstm_ner(max_len_sent, embedding_dims, num_tags):
    
    X_in = Input(shape = (max_len_sent,))
    X = Embedding(input_dim = embedding_dims[0], 
                   output_dim = embedding_dims[1], 
                   input_length = max_len_sent, 
                   mask_zero = False, 
                   name = 'embedding_layer')(X_in)
    
    X = Bidirectional(LSTM(max_len_sent, dropout = 0.3, return_sequences = True))(X)
    X = TimeDistributed(Dense(num_tags, activation = 'softmax'))(X)

    model = Model(inputs = X_in, outputs = X)
    return model



