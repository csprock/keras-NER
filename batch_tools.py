import numpy as np

def indices_by_length(data):
    '''
    Groups sample indices by sentence length. Input is a list of dictionaries with key 'length'.
    Returns dictionary keyed to sentence length containing a list of indices. 
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
    '''
    n_idx, n_batches = len(indices), len(indices)//batch_size
    batches = dict()

    if n_idx % batch_size == 0:
        for i in range(n_batches):
            if shuffle:
                batches[i + start_index] = np.random.permutation(indices[(i*batch_size):((i+1)*batch_size)])
            else:
                batches[i + start_index] = indices[(i*batch_size):((i+1)*batch_size)]
    else:
        for i in range(n_batches+1):
            if shuffle:
                batches[i + start_index] = np.random.permutation(indices[(i*batch_size):((i+1)*batch_size)])
            else:
                batches[i + start_index] = indices[(i*batch_size):((i+1)*batch_size)]
            
    return batches

def create_same_length_batches(length_sorted_indices, batch_size, separate = True, shuffle = False):
    '''
    Takes a dictionary of indices and converts them into dictionaries of batches using
    batch_dict().
    '''
    accum = 0
    batches = dict()
    if separate:
        
        for l, idx in length_sorted_indices.items():
            batches[l] = batch_dict(idx, batch_size = batch_size, start_index = accum)
            accum += len(batches[l])        
    else:
        
        for l, idx in length_sorted_indices.items():
            new_batch = batch_dict(idx, batch_size = batch_size, start_index = accum)
            accum += len(new_batch)
            batches.update(new_batch)
        
        if shuffle:
                
            new_arrangement = np.random.permutation(list(batches.keys()))
            
            for k, v in batches.items():
                i = new_arrangement[k]
                batches[k], batches[i] = batches[i], v
        
    return batches

def create_batch_indices(data, batch_size, separate = True, shuffle = False):
    return create_same_length_batches(indices_by_length(data), batch_size = batch_size, separate = separate, shuffle = shuffle)



        


