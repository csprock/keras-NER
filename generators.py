import numpy as np
import keras
from batch_utils import create_batch_indices
import random


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, data, tensor_maker, batch_size, shuffle, sentences, characters, word_features, tags):
        
        # data type parameters to be passed to TensorMaker.makeTensors() method
        self.sentences = sentences
        self.characters = characters
        self.word_features = word_features
        self.tags = tags
        # TensorMaker object
        self.tensor_maker = tensor_maker
        # data
        self.data = data
        # parameters for batch creation and initial batch indices
        self.batch_size = batch_size
        self.indices = create_batch_indices(data, batch_size, shuffle)
        self.shuffle = shuffle
        # internal method for keras
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.indices)
    
    def __data_generation(self, index_list):
        
        
        temp_sent = [self.data[i] for i in index_list]
        L = len(temp_sent[0]['sentence'])
        # X_sent, X_char, X_word_ft, Y
        
        return self.tensor_maker.makeTensors(temp_sent, L, 
                                             sentences = self.sentences, 
                                             characters = self.characters, 
                                             word_features = self.word_features, 
                                             tags = self.tags)
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
        
        # initialize TensorMaker object and data
        self.data = data
        self.tensor_maker = tensor_maker
        
        # create initial indices
        self.indices = create_batch_indices(data, batch_size)
        
        # parameters to be passed to TensorMaker.makeTensors()
        self.sentences = sentences
        self.characters = characters
        self.word_features = word_features
        self.tags = tags
        
        # initialize number of batches
        self.I = len(self.indices)
       
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.I == 0:
            raise StopIteration
        
        # decrement remaining batches
        self.I -= 1
        
        temp_sent = [self.data[i] for i in self.indices[self.I]]
        L = len(temp_sent[0]['sentence'])
        
        X_data, Y = self.tensor_maker.makeTensors(temp_sent, L, 
                                                  sentences = self.sentences, 
                                                  characters = self.characters, 
                                                  word_features = self.word_features, 
                                                  tags = self.tags)
        
        return X_data, Y
    
    
    
class CharDataGenerator(keras.utils.Sequence):
    
    def __init__(self, data, tensor_maker, batch_size, shuffle, char_features, tags):
        
        # data type parameters to be passed to TensorMaker.makeTensors() method
        self.char_features = char_features
        self.tags = tags
        # TensorMaker object
        self.tensor_maker = tensor_maker
        # data
        self.data = data
        # parameters for batch creation and initial batch indices
        self.batch_size = batch_size
        self.indices = create_batch_indices(data, batch_size, shuffle)
        self.shuffle = shuffle
        # internal method for keras
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.indices)
    
    def __data_generation(self, index_list):
        
        
        temp_sent = [self.data[i] for i in index_list]
        # X_sent, X_char, X_word_ft, Y
        
        return self.tensor_maker.makeTensors(temp_sent, 
                                             char_features = self.char_features, 
                                             tags = self.tags)
        #return {'word_input':X_sent, 'word_feature_input':X_word_ft, 'char_input':X_char}, Y
        #return X_data, Y
    
        
    def __getitem__(self, index):
        
        return self.__data_generation(index_list = self.indices[index])
        #return X, y
        
    def on_epoch_end(self):
        # re-initialize indices upon end of epoch
        self.indices = create_batch_indices(self.data, self.batch_size, shuffle = self.shuffle)
        
        
class CharTestDataGenerator:
    
    def __init__(self, data, tensor_maker, batch_size, char_features, tags):
        
        # initialize TensorMaker object and data
        self.data = data
        self.tensor_maker = tensor_maker
        
        # create initial indices
        self.indices = create_batch_indices(data, batch_size)
        
        # parameters to be passed to TensorMaker.makeTensors()
        self.char_features = char_features
        self.tags = tags
        
        # initialize number of batches
        self.I = len(self.indices)
       
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.I == 0:
            raise StopIteration
        
        # decrement remaining batches
        self.I -= 1
        
        temp_sent = [self.data[i] for i in self.indices[self.I]]
        
        #### need to modify to handle then tags = False
        X_data, Y = self.tensor_maker.makeTensors(temp_sent, 
                                             char_features = self.char_features, 
                                             tags = self.tags)
        
        return X_data, Y
    
    

## Hyperparameters
# learning rate
# recurrent dropout rate
# dropout rate of main FC
# dropout rate of character-level part
# char embedding dimension
# number of neurons in FC layers
# batch size


def random_hyperparameters():
        
    lr = 10**random.uniform(-5, -0.5)
    recurrent_dropout = random.uniform(0.25, 1)
    main_dropout = random.uniform(0.25, 1)
    fc_cnn_dropout = random.uniform(0.5, 1)
    batch_size = random.choice([2**i for i in [4,5,6,7,8]])
    char_embedding = random.choice([4,8,12,16,20])
    cnn_filters = random.randrange(25, 75)