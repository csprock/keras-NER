import numpy as np
import keras
from batch_tools import create_batch_indices


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, batch_size, tensor_maker, data, shuffle, sentences, characters, word_features, tags):
        
        # data type parameters to be passed to TensorMaker.makeTensors() method
        self.sentences = sentences
        self.characters = characters
        self.word_features = word_features
        self.tags = tags
        
        self.batch_size = batch_size
        self.indices = create_batch_indices(data, batch_size, shuffle)
        self.tensor_maker = tensor_maker
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.indices)
    
    def __data_generation(self, index_list):
        
        
        temp_sent = [self.data[i] for i in index_list]
        L = len(temp_sent[0]['sentence'])
        # X_sent, X_char, X_word_ft, Y
        
        return self.tensor_maker.makeTensors(temp_sent, L, sentences = self.sentences, characters = self.characters, word_features = self.word_features, tags = self.tags)
        #return {'word_input':X_sent, 'word_feature_input':X_word_ft, 'char_input':X_char}, Y
        #return X_data, Y
    
        
    def __getitem__(self, index):
        
        return self.__data_generation(index_list = self.indices[index])
        #return X, y
        
    def on_epoch_end(self):
        # re-initialize indices upon end of epoch
        self.indices = create_batch_indices(self.data, self.batch_size, shuffle = self.shuffle)
        
        
class TestDataGenerator:
    
    def __init__(self, data, batch_size, tensor_maker, sentences, characters, word_features, tags):
        
        self.data = data
        self.tensor_maker = tensor_maker
        
        self.indices = create_batch_indices(data, batch_size)
        self.sentences = sentences
        self.characters = characters
        self.word_features = word_features
        self.tags = tags
        
        self.I = len(self.indices)
       
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.I == 0:
            raise StopIteration
            
        self.I -= 1
        
        temp_sent = [self.data[i] for i in self.indices[self.I]]
        L = len(temp_sent[0]['sentence'])
        
        X_data, Y = self.tensor_maker.makeTensors(temp_sent, L, sentences = self.sentences, characters = self.characters, word_features = self.word_features, tags = self.tags)
        
        return X_data, Y