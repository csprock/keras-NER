import pandas as pd

def parse_line(line):
    
    split_line = line.split(' ')
    split_line[3] = split_line[3].replace('\n', '')
    
    word, pos, chunk, tag = split_line[0], split_line[1], split_line[2], split_line[3]
    
    return word, pos, chunk, tag


def conll_to_df(path, start_index = None):
    '''
    Convert CoNLL2003 formatted data into pandas dataframe with the 4 CoNLL columns plus an additional
    label column where the label is index increasing from start_index if given or 1 if not given. 
    '''
    f = open(path)
    r = f.readlines()
    
    doc = list()
    current_sentence = list()
    
    if start_index is None: i = 1
    else: i = start_index
    
    for line in r:
        
        sent_label = 'Sentence: {}'.format(i)
        if len(line) > 1:
            
            if 'DOCSTART' not in line:
                word, pos, chunk, tag = parse_line(line)
                current_sentence.append([sent_label, word, pos, chunk, tag])
                
        else:
            doc.extend(current_sentence)
            i += 1
            current_sentence = list()
    
    return pd.DataFrame(doc, columns = ['Sentence #','Word','POS','Chunk','Tag'])
                                        
