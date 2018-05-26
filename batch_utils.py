import numpy as np

def indices_by_length(data):
    '''
    Groups sample indices by sentence length. Input is a list of dictionaries with each 
    dictionary having a key 'length'. Returns dictionary keyed to sentence length containing a list of indices. 
    
    Input
    -----
    data:
        list of data dictionaries with a key called 'length'
    
    Returns
    -------
    dict: dictionary whose keys are integers and whose values are lists of integers.
        Dictionary keys are lengths of sentences, values are lists of indices in the data set
    '''
    
    length_batches = dict()
    for i, d in data.items():
        if d['length'] not in length_batches:
            length_batches[d['length']] = [i]
        else:
            length_batches[d['length']].append(i)
            
    return length_batches
        
def batch_dict(indices, batch_size, start_index = 0, shuffle = False):
    '''
    Turns the list of integers 'indices' into batches of size 'batch_size' and returns a 
    dictionary keyed to the order in which they were processed plus 'start_index'.
    
    Input
    -----
    indices: list[int]
        list of integers
    batch_size: int
        size of batches
    start_index: int
        offset on the batch number
    shuffle: bool
        permute indices within batch
        
    Returns
    -------
    dict: (int, list)
        dictionary where keys are batch numbers and values are list of data indices
    '''
    
    n_idx, n_batches = len(indices), len(indices)//batch_size
    batches = dict()

    if n_idx % batch_size == 0:
        for i in range(n_batches):
            
            batch = indices[(i*batch_size):((i+1)*batch_size)]
            
            if shuffle:
                batches[i + start_index] = np.random.permutation(batch)
            else:
                batches[i + start_index] = batch
    else:
        for i in range(n_batches+1):
            
            batch = indices[(i*batch_size):((i+1)*batch_size)]
            
            if shuffle:
                batches[i + start_index] = np.random.permutation(batch)
            else:
                batches[i + start_index] = batch
            
    return batches

def create_same_length_batches(length_sorted_indices, batch_size, shuffle = False):
    '''
    Takes a dictionary of indices and converts them into dictionaries of batches using
    batch_dict().
    
    Input
    -----
    length_sorted_indices: dict (int, list)
        dictionary where keys are integers representing sentence lengths and values are lists of data indices
        which is returned by the indices_by_length()
    batch_size: int
    shuffle: bool
        shuffle order of the batches
        
    Returns
    -------
    batches: dict
        dictionary whose keys are batch indices and whose values are a list of indices
    
    '''
    accum = 0
    batches = dict()
    
    # creates batches 
    for l, idx in length_sorted_indices.items():
        new_batch = batch_dict(idx, batch_size = batch_size, start_index = accum)
        batches.update(new_batch)
        
        accum += len(new_batch)
    
    # shuffles the order of batches
    if shuffle:
            
        new_arrangement = np.random.permutation(list(batches.keys()))
        
        for k, v in batches.items():
            i = new_arrangement[k]
            batches[k], batches[i] = batches[i], v
        
    return batches

def create_batch_indices(data, batch_size, shuffle = True):
    ''' Create dictionary of batches '''
    return create_same_length_batches(indices_by_length(data), 
                                      batch_size = batch_size, 
                                      shuffle = shuffle)

