import numpy as np
from keras.preprocessing.sequence import pad_sequences


##################################################
#########  index dictionary makers ###############
##################################################
def create_character_dictionary(vocab_list, custom = False):
    '''
    Accepts a list of words and creates lookup dictionary containing each character found in 
    the set of words. Adds two additional tokens for 'UNK' and 'PAD'
    '''
    
    if custom:
        assert isinstance(vocab_list, str)
        char2idx = {c: i+2 for i, c in enumerate(vocab_list)}
    else:
        i = 2
        char2idx = dict()
        for w in vocab_list:
            for c in w:
                if c not in char2idx:
                    char2idx[c] = i
                    i += 1
        
    char2idx['UNK'], char2idx['PAD'] = 1, 0
    return char2idx

        

def create_word_dictionary(vocab_list):
    '''
    Accepts a list of words and creates a lookup dictionary containing each word found in vocab_list. 
    Assumes the words in vocab_list are unique. Adds additional tokens for 'UNK' and 'PAD'
    '''
    word2idx = {w: i+2 for i, w in enumerate(vocab_list)}
    word2idx['UNK'], word2idx['PAD'] = 1, 0
    return word2idx



def create_tag_dictionary(tag_list, zero_tag = None):
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
    
def word_features(word):
    '''
    Accepts a string. Return feature indicator for input word. 
    
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
    

def char_features(char, punctuation_list = '\'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    '''
    Accepts a str of length 1.
    
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


####### sequence converters ###########

def sent2seq(sent, word_dictionary):
    '''
    input sentence as list of words and a word index dictionary
    '''
    sent_seq = [None]*len(sent)
    for i, w in enumerate(sent):
        widx = word_dictionary.get(w.lower())
        if widx is not None:
            sent_seq[i] = widx
        else:
            sent_seq[i] = word_dictionary['UNK']
    
    return sent_seq



def word2seq(word, char_dictionary):
    '''
    convert word to sequence of integers given word and character index dictionary
    assumes words have been stripped of bordering whitespace
    '''
    char_seq = [None]*len(word)
    for i, c in enumerate(word):
        cidx = char_dictionary.get(c.lower())
        if cidx is not None:
            char_seq[i] = cidx
        else:
            char_seq[i] = char_dictionary['UNK']
    
    return char_seq

def tags2seq(tags, tag_dictionary):
    return [tag_dictionary[t] for t in tags]



def char_feature_seq(word):
    '''
    Converts word into sequence of character features of same length.
    
    Requires: char_features()
    '''
    
    char_feat_seq = [None]*len(word)
    for i, c in enumerate(word): char_feat_seq[i] = char_features(c)
    return char_feat_seq

def word_feature_seq(sent):
    '''
    Converts sentence into sequence of word features of same length.
    
    Requires: word_features()
    '''
    word_feat_seq = [None]*len(sent)
    for i, w in enumerate(sent): word_feat_seq[i] = word_features(w)
    return word_feat_seq


##### character tensor slices #######


def char_matrix(sentence, char2idx, max_len_word, max_len_sent, word_padding, word_truncating, sent_padding):
    '''
    Create a matrix of dimension (max_len_sent, max_len_word). Each row is a sequence of character indices representing
    that word. 
    
    Requires: word2seq(), pad_sequences() from keras.preprocessing.sequence
    
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
    
    X = pad_sequences([word2seq(w, char2idx) for w in sentence], 
                       maxlen = max_len_word, 
                       padding = word_padding,
                       truncating = word_truncating)
    
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
        


def char_feature_matrix(sentence, max_len_word, max_len_sent, word_padding, word_truncating, sent_padding):
    '''
    Create a matrix of dimension (max_len_sent, max_len_word). Each row is a sequence of feature indices the length
    of that word. 
    
    Requires: char_feature_seq(), pad_sequences() from keras.preprocessing.sequence
    
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
        array of dimension (max_len_sent, max_len_word)
    '''
    X = pad_sequences([char_feature_seq(w) for w in sentence], 
                       maxlen = max_len_word, 
                       padding = word_padding,
                       truncating = word_truncating)
    
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



#### configuration variables ####
# word_padding
# word_truncating
# send_padding
# max_len_sent
# max_len_word
# zero_tag
        
################ feature transformer classes ####################### 

class sentenceTransformer:
    
    def __init__(self, word_list, max_len_sent, max_len_word, sent_padding, sent_truncating, word_padding, word_truncating):
        
        
        self.word2idx = create_word_dictionary(word_list)
        self.char2idx = create_character_dictionary(word_list)
        
        self.word_padding = word_padding
        self.word_truncating = word_truncating
        self.sent_padding = sent_padding
        self.sent_truncating = sent_truncating
        self.max_len_sent = max_len_sent
        self.max_len_word = max_len_word
        
    
    def wordSequence(self, sentence):
        return sent2seq(sentence, self.word2idx)
    
    def wordFeatures(self, sentence):
        return word_feature_seq(sentence)
    
    def charSequence(self, sentence):
        return char_matrix(sentence, 
                           self.char2idx,
                           self.max_len_word, 
                           self.max_len_sent,
                           self.word_padding,
                           self.word_truncating,
                           self.sent_padding)
        
    def charFeatures(self, sentence):
        return char_feature_matrix(sentence, 
                           self.max_len_word, 
                           self.max_len_sent,
                           self.word_padding,
                           self.word_truncating,
                           self.sent_padding)
        
        
    def pad_sentences(self, sentence_list):
        '''
        takes list of sentences after wordSequence() has been called
        '''
        return pad_sequences(sentence_list, 
                             padding = self.sent_padding, 
                             truncating = self.sent_truncating, 
                             maxlen = self.max_len_sent, 
                             value = 0)


class tagTransformer:
    
    def __init__(self, tag_list, pad_value, max_len_sent, padding, truncating, zero_tag = None):
        assert pad_value in tag_list
        self.tag2idx = create_tag_dictionary(tag_list, zero_tag)
        self.pad_value = pad_value
        self.zero_tag = zero_tag
        self.max_len_sent = max_len_sent
        self.padding = padding
        self.truncating = truncating
        
    def tagSequence(self, tags):
        return tags2seq(tags, self.tag2idx)
       
    def pad_tags(self, tag_sequences):
        return pad_sequences(tag_sequences, 
                             maxlen = self.max_len_sent, 
                             padding = self.padding,
                             truncating = self.truncating,
                             value = self.tag2idx[self.pad_value])
        




def parse_sentence(i, data, inputs_only = False):
    ''' return tuple (word, tag) or (word, None)'''
    sent = data.loc[data['Sentence #'] == 'Sentence: {}'.format(i)]
    if inputs_only:
        return list(sent['Word']), None
    else:
        return list(sent['Word']), list(sent['Tag'])



