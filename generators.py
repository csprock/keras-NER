import numpy as np
import keras
from batch_tools import create_batch_indices
# __getitem__
# __len__
# __data_generation
# on_epoch_end


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, batch_size, tensor_maker, data, shuffle):
        
        self.batch_size = batch_size
        self.indices = create_batch_indices(data, batch_size, separate = False, shuffle = shuffle)
        self.tensor_maker = tensor_maker
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.indices)
    
    def __data_generation(self, index_list):
        
        def get_sentences(indices, data):
            return [data[i] for i in indices]    
    
        temp_sent = get_sentences(index_list, self.data)
        L = len(temp_sent[0]['sentence'])
        # X_sent, X_char, X_word_ft, Y
        X_sent, X_char, X_word_ft, Y = self.tensor_maker.makeTensors(temp_sent, L, sentences = True, characters = True, word_features = True, tags = True)
        return {'word_input':X_sent, 'word_feature_input':X_word_ft, 'char_input':X_char}, Y
    
        
    def __getitem__(self, index):
        
        X, y = self.__data_generation(index_list = self.indices[index])
        return X, y
        
    def on_epoch_end(self):
        self.indices = create_batch_indices(self.data, self.batch_size, separate = False, shuffle = self.shuffle)
        
        
