import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

##################################################
#########  index dictionary makers ###############
##################################################
def _create_character_dictionary(vocab_list, whitespace = False, custom = False):
    '''
    Accepts a list of words and creates lookup dictionary containing each character found in 
    the set of words or accepts a string of characters to be used in the lookup table. 
    Adds two additional tokens for 'UNK' and 'PAD' and optional whitespace token which 
    can be by setting whitespace = True. 
    
    Characters are first lowercased before being added to dictionary.
    
    The setting the custom parameter to True will treat vocab_list as a sequence of characters
    to be converted directly into the character_dictionary.
    '''
    # start dictionary index at 2 or 3 since 0 and 1 reserved
    if custom:
        
        assert isinstance(vocab_list, str)
        
        if whitespace: char2idx = {c: i+3 for i, c in enumerate(vocab_list)}
        else: char2idx = {c: i+2 for i, c in enumerate(vocab_list)}
    else:
        
        assert isinstance(vocab_list, list)
        
        if whitespace: i = 3
        else: i = 2
        
        char2idx = dict()
        for w in vocab_list:
            for c in w:
                if c.lower() not in char2idx:
                    char2idx[c.lower()] = i
                    i += 1
        
    char2idx['UNK'], char2idx['PAD'] = 1, 0
    if whitespace: char2idx[' '] = 2
    
    return char2idx

        

def _create_word_dictionary(vocab_list):
    '''
    Accepts a list of words and creates a lookup dictionary containing each word found in vocab_list. 
    Assumes the words in vocab_list are unique. Adds additional tokens for 'UNK' and 'PAD'. Words
    are lower cased before added to dictionary.
    '''
    word2idx = {w.lower(): i+2 for i, w in enumerate(vocab_list)}
    word2idx['UNK'], word2idx['PAD'] = 1, 0
    return word2idx



def _create_tag_dictionary(tag_list, zero_tag = None):
    '''
    Accepts a list of tags and returns a lookup dictionary containing each tag. Optional
    zero_tag to specify which tag to assign an index of 0.
    '''
    tag2idx = {t: i for i, t in enumerate(tag_list)}
    
    if zero_tag is not None:
        assert zero_tag in tag2idx
        for k, v in tag2idx.items():
            if v == 0:
                tag2idx[k] = tag2idx[zero_tag]
                tag2idx[zero_tag] = 0
        
            
    return tag2idx

################################################
########## word and char features  #############
################################################
    
def _word_features(word):
    '''
    Accepts a string. Return feature index/indicator for input word. 
    
    Features:
        - all lowercase
        - all uppercase
        - capitalized
        - mixed upper and lower case
        - other
        
    '''
    
    if word.islower():
        return 1
    elif word.isupper():
        return 2
    elif word.istitle():
        return 3
    elif any([c.isupper() for c in word]):
        return 4
    else:
        return 5
    

def _char_features(char, punctuation_list = '\'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    '''
    Accepts a str of length 1. Returns feature index/indicator for input character.
    
    Features:
        - uppercase
        - lowercase
        - punctuation
        - other
    '''
                  
    if char.isupper():
        return 1
    elif char.islower():
        return 2
    elif char in punctuation_list:
        return 3
    else:
        return 4


##################################################
#########  convert sequences       ###############
##################################################
                
def _sent2seq(sent, word_dictionary):
    '''
    Converts sentence into a sequence of indices using a word_dictionary. Sentences are list of 
    word tokens. 
    
    Input
    -----
    sent: list of str
        each str is a word in sentence
    word_dictionary: dict
        dictionary containing mappings from words to indices
        
    Output
    ------
    sent_seq: list of int
    '''
    sent_seq = [None]*len(sent)
    for i, w in enumerate(sent):
        widx = word_dictionary.get(w.lower())
        if widx is not None:
            sent_seq[i] = widx
        else:
            sent_seq[i] = word_dictionary['UNK']
    
    return sent_seq

def _word2seq(word, char_dictionary):
    '''
    Convert word into sequence of indices using char_dictionary.
    
    Input
    -----
    word: str
    char_dictionary: dict
        dictionary containing mappings from characters to indices
    
    Output
    ------
    char_seq: list of int
    '''
    char_seq = [None]*len(word)
    for i, c in enumerate(word):
        cidx = char_dictionary.get(c.lower())
        if cidx is not None:
            char_seq[i] = cidx
        else:
            char_seq[i] = char_dictionary['UNK']
    
    return char_seq

def _tags2seq(tags, tag_dictionary):
    return [tag_dictionary[t] for t in tags]

def _char_feature_seq(word):
    '''
    Converts word into sequence of character feature (represented as integers) of same length.
    
    Requires: _char_features()
    '''
    
    char_feat_seq = [None]*len(word)
    for i, c in enumerate(word): char_feat_seq[i] = _char_features(c)
    return char_feat_seq

def _word_feature_seq(sent):
    '''
    Converts sentence into sequence of word features (represented as integers) of same length.
    
    Requires: _word_features()
    '''
    word_feat_seq = [None]*len(sent)
    for i, w in enumerate(sent): word_feat_seq[i] = _word_features(w)
    return word_feat_seq

###########################################
##### character-level tensor slices #######
###########################################
    
def _entity_matrix(entities, char2idx, max_len_word, word_padding, word_truncating):
    '''
    This function takes a list of entities (strings) and returns a numpy array of dimension
    (number of entities) x (max_len_word). 
    
    Requires: _word2seq()
    
    Input
    -----
    entities: list of str
        list of strings representing words
    char2idx: dict
        dictionary containing mappings from characters to indices
    max_len_word: int
        maximum length of entity
    word_padding: str
        'post' for padding after end of entity, 'pre' for padding before
    word_truncating: str
        'post' for truncating ends of words whose length exceeds max_len_word, 
        'pre' for truncating beginnings
        
    Output
    ------
    X: numpy array of dimention (len(entities), max_len_word)
    '''
    X = pad_sequences([_word2seq(ne, char2idx) for ne in entities],
                       maxlen = max_len_word,
                       padding = word_padding,
                       truncating = word_truncating)
    return X
    

def _entity_feature_matrix(entities, char2idx, max_len_word, word_padding, word_truncating):
    
    X = pad_sequences([_char_feature_seq(ne) for ne in entities],
                      maxlen = max_len_word,
                      padding = word_padding, 
                      truncating = word_truncating)
    return X

def _char_matrix(sentence, char2idx, max_len_word, max_len_sent, word_padding, word_truncating, sent_padding, matrix_padding = True):
    '''
    Create a matrix of dimension (max_len_sent, max_len_word). Each row is a sequence of character indices representing
    that word. 
    
    Requires: _word2seq(), pad_sequences() from keras.preprocessing.sequence
    
    Inputs
    ------
    sentence: list of str
    char2idx: dic
    max_len_word: int
    max_len_sent: int
    word_padding: str 'post' or 'pre'
        pad words to max_len_word before or after
    word_truncating: str 'post' or 'pre'
        truncate words to max_len_word before or after
    sent_padding: str 'post' or 'pre'
        pad matrix along sentence dimention with zeros before or after
        
    Returns
    -------
    X: numpy array
        array of dimension (max_len_sent, max_len_word)
    '''
    
    X = pad_sequences([_word2seq(w, char2idx) for w in sentence], 
                       maxlen = max_len_word, 
                       padding = word_padding,
                       truncating = word_truncating)
    
    if matrix_padding:
        if X.shape[0] < max_len_sent:
            # padding matrix with excess dimensions
            X_pad = np.zeros((max_len_sent - X.shape[0], max_len_word))
            
            # concatenate padding matrix and character matrix
            if sent_padding == 'post':
                X = np.concatenate((X, X_pad))
            elif sent_padding == 'pre':
                X = np.concatenate((X_pad, X))
            else:
                raise ValueError
            
            return X
        
        elif X.shape[0] > max_len_sent:
            if sent_padding == 'post':
                X = X[:max_len_sent,:]
            elif sent_padding == 'pre':
                X = X[max_len_sent:,:]
            else:
                raise ValueError
                
            return X
        else:
            return X
    else:
        return X
        


def _char_feature_matrix(sentence, max_len_word, max_len_sent, word_padding, word_truncating, sent_padding, matrix_padding = True):
    '''
    Create a matrix of dimension (max_len_sent, max_len_word). Each row is a sequence of feature indices the length
    of that word. 
    
    Requires: _char_feature_seq(), pad_sequences() from keras.preprocessing.sequence
    
    Inputs
    ------
    sentence: list of str
    max_len_word: int
    max_len_sent: int
    word_padding: str 'post' or 'pre'
        pad words to max_len_word before or after
    word_truncating: str 'post' or 'pre'
        truncate words to max_len_word before or after
    sent_padding: str 'post' or 'pre'
        pad matrix along sentence dimention with zeros before or after
        
    Returns
    -------
    X: numpy array
        array of dimension (max_len_sent, max_len_word) if matrix_padding = True
        array of dimension (len(sentence), max_len_word) if matrix_padding = False
    '''
    X = pad_sequences([_char_feature_seq(w) for w in sentence], 
                       maxlen = max_len_word, 
                       padding = word_padding,
                       truncating = word_truncating)
    
    if matrix_padding:
        if X.shape[0] < max_len_sent:
            X_pad = np.zeros((max_len_sent - X.shape[0], max_len_word))   # pad extra dimensions with 0's
            
            if sent_padding == 'post':
                X = np.concatenate((X, X_pad))
            elif sent_padding == 'pre':
                X = np.concatenate((X_pad, X))
            else:
                raise ValueError
            
            return X
        elif X.shape[0] > max_len_sent:
            if sent_padding == 'post':
                X = X[:max_len_sent,:]
            elif sent_padding == 'pre':
                X = X[max_len_sent:,:]
            else:
                raise ValueError
                
            return X
        else:
            return X
    else:
        return X

###################################################
#########  transformer classes      ###############
###################################################

class CharTensorMaker:
    
    def __init__(self, entity_list, tag_list, max_len_word, word_padding = 'post', word_truncating = 'post', zero_tag = None, custom_chars = False):
        
        self.tag2idx = _create_tag_dictionary(tag_list, zero_tag = zero_tag)
        self.char2idx = _create_character_dictionary(entity_list, custom = custom_chars)
        
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.word_padding = word_padding
        self.word_truncating = word_truncating
        self.max_len_word = max_len_word
        
    def charSequence(self, entities, ne_length):
        return _entity_matrix(entities,
                              self.char2idx,
                              self.max_len_word,
                              #ne_length,
                              self.word_padding,
                              self.word_truncating)
        
    def charFeatures(self, entities, ne_length):
        return _entity_feature_matrix(entities, 
                                      self.char2idx,
                                      self.max_len_word, 
                                      #ne_length,
                                      self.word_padding,
                                      self.word_truncating)
        
    def idx2tags(self, Y):
        '''
        Takes in a batch x 1 dimensional numpy array (the result of which is
        an np.argmax on the predicted output of the model)
        '''
        return [self.idx2tag[i] for i in list(Y)]
        
    def makeTensors(self, data, characters = True, char_features = False, tags = True):
        
        
        outputs = dict()
        entities = list()
        
        for d in data:
            entities.append(d['entity'])
        
        L = len(entities[0])
        
        if characters:
            outputs['X_char'] = self.charSequence(entities, L)
        if char_features:
            outputs['X_char_feat'] = self.charFeatures(entities, L)
        if tags:
            tag_list = list()
            for d in data:
                tag_list.append(d['label'])
            
            Y = np.asarray(_tags2seq(tag_list, self.tag2idx))
            Y = to_categorical(Y, num_classes = len(self.tag2idx))
        
        if tags:
            return outputs, Y
        else:
            return outputs

class TensorMaker:

    def __init__(self, word_list, tag_list, max_len_word, word_padding = 'post', word_truncating = 'post', matrix_padding = False, zero_tag = None, custom_chars = False):
        
        self.tag2idx = _create_tag_dictionary(tag_list, zero_tag = zero_tag)
        self.word2idx = _create_word_dictionary(word_list)
        self.char2idx = _create_character_dictionary(word_list, custom = custom_chars)
        
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        #self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.matrix_padding = matrix_padding
        self.word_padding = word_padding
        self.word_truncating = word_truncating
        self.max_len_word = max_len_word
        
        
    @property
    def classes(self):
        return len(self.tag2idx)
    
    @property
    def vocab_size(self):
        return len(self.word2idx)
    
    @property
    def char_vocab_size(self):
        return len(self.char2idx)
        
    def tagSequence(self, tags):
        ''' Wrapper for _tags2seq()'''
        return _tags2seq(tags, self.tag2idx)
    
        
    def wordSequence(self, sentence):
        ''' Wrapper for _sent2seq() '''
        return _sent2seq(sentence, self.word2idx)
    
    def wordFeatures(self, sentence):
        ''' Wrapper for _word_feature_seq()'''
        return _word_feature_seq(sentence)
    
    def charSequence(self, sentence):
        ''' Wrapper for _char_matrix() '''
        return _char_matrix(sentence, 
                           self.char2idx,
                           self.max_len_word, 
                           None, # max_len_sent
                           self.word_padding,
                           self.word_truncating,
                           None, # sent_padding
                           self.matrix_padding)
        
    def charFeatures(self, sentence):
        ''' Wrapper for _char_feature_matrix() '''
        return _char_feature_matrix(sentence, 
                           self.max_len_word, 
                           None, # max_len_sent
                           self.word_padding,
                           self.word_truncating,
                           None, # sent_padding
                           self.matrix_padding)
        
        
        
    def convert2tags(self, sentence):
        '''
        Convert sequence of indices back to tags.
        '''
        return [self.idx2tag[t] for t in sentence]
    
    def convert2words(self, sentence):
        '''
        Convert sequence of indices back to words.
        '''
        return [self.idx2word[t] for t in sentence]
        
    def makeTensors(self, data, sent_len, sentences = True, characters = False, word_features = False, tags = False, lists = False):
        '''
        Main accessor method for converting sequences of tokens into tensors. Is a wrapper for the other 
        methods of the TensorMaker class. 
        
        Inputs
        ------
        data: list of dict
            List of dictionaries containing keys 'sentence' and 'tags', each containing a list of tokens to be converted.
            If tags set to False, dictionaries do not need to contain the key 'tags'. 
        sentences: bool
            return sentence tensors
        characters: bool
            return character tensors
        word_features: bool
            return word feature tensors
        tags: bool
            return tag tensors (for training)
        
        '''
        N = len(data)
        
        X_sent, X_char, X_word_ft, Y = [None]*N, [None]*N, [None]*N, [None]*N
        X_outputs = dict()
        
        for i, s in enumerate(data):
                                
            if sentences:
                X_sent[i] = self.wordSequence(s['sentence'])
            if characters:
                X_char[i] = self.charSequence(s['sentence'])
            if word_features:
                X_word_ft[i] = self.wordFeatures(s['sentence'])
            if tags:
                Y[i] = self.tagSequence(s['tags'])
                
        if sentences:
            X_sent = np.asarray(X_sent)
            X_outputs['word_input'] = X_sent
        if characters:
            X_char = np.dstack(X_char).reshape((N, sent_len, self.max_len_word))
            X_outputs['char_input'] = X_char
        if word_features:
            X_word_ft = np.asarray(X_word_ft)
            X_outputs['word_feature_input'] = X_word_ft
        if tags:
            Y = np.asarray(Y)
            Y = np.array([to_categorical(i, len(self.tag2idx)) for i in Y])
            
        if tags:
            return X_outputs, Y
        else:
            return X_outputs


def parse_sentence(i, data, inputs_only = False):
    ''' return tuple (word, tag) or (word, None)'''
    sent = data.loc[data['Sentence #'] == 'Sentence: {}'.format(i)]
    if inputs_only:
        return list(sent['Word']), None
    else:
        return list(sent['Word']), list(sent['Tag'])



