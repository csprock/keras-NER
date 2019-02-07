
#A = 'xxxxxxxxxbixxxxbxb'
#B = 'xxbxxxxxxbixxxxxxb'
#
#n = len(A)
#i = 0
#cp, tp = 0, 0
#
#while i < n:
#    
#    if A[i] == 'b':
#        cp += 1
#        if A[i] == B[i]:
#            Found = True
#            while i < n and A[i] in ['b','i']:
#                if A[i] != B[i]:
#                    Found = False
#                i += 1
#            
#            if i < n:
#                if B[i] == 'i':
#                    Found = False
#            
#            if Found:
#                tp += 1
#                
#        else:
#            i += 1
#    else:
#        i += 1
#
#
#print("tp = %s, cp = %s" % (tp, cp))
#
#def sentence_metrics2(predicted, actual):
#    assert len(predicted) == len(actual)
#    n = len(predicted)
#    i = 0
#    cp, tp = 0, 0
#    
#    while i < n:
#        
#        if actual[i][0] == 'B':
#            cp += 1
#            if actual[i][0] == predicted[i][0]:
#                Found = True
#                while i < n and actual[i][0] in ['B','I']:
#                    if actual[i][0] != predicted[i][0]:
#                        Found = False
#                    i += 1
#                
#                if i < n:
#                    if predicted[i][0] == 'I':
#                        Found = False
#                
#                if Found:
#                    tp += 1
#                    
#            else:
#                i += 1
#        else:
#            i += 1
#            
#            
#    return tp, cp
#
#
###################
#
#A = 'xxxxxb'
#B = 'xxxxxb'
#n = len(A)
#cp = 0
#start_ne = list()
#for i, s in enumerate(A):
#    if s == 'b':
#        cp += 1
#        start_ne.append(i)
#
#tp = 0
#
#for start_i in start_ne:
#    
#    i = start_i
#    while i < n and A[i] in ['b','i']:
#        
#        if A[i] == B[i]:
#            found = True
#        else:
#            found = False
#        i += 1
#        
#    if i < n and found:
#        if A[i] == B[i]:
#            tp += 1
#    elif found:
#        tp += 1
#    else:
#        pass
#    
#print("tp = %s, cp = %s" % (tp, cp))



def sentence_metrics(predicted, actual):
    '''
    Return number of true positives (tp) and actual positives(cp) in between
    a predicted and target sentence.
    '''
    cp, tp = 0, 0
    n = len(predicted)
    start_ne = list()
    for i, s in enumerate(actual):
        if s[0] == 'B':
            cp += 1
            start_ne.append(i)
    
    
    for start_i in start_ne:
        
        i = start_i
        while i < n and actual[i][0] in ['B','I']:
            
            if actual[i][0] == predicted[i][0]:
                # found = True
                
                try:
                    if actual[i].split('-')[1] == predicted[i].split('-')[1]:
                        found = True
                    else:
                        found = False
                except IndexError:
                    found = False
                    
            else:
                found = False
            i += 1
            
            
        if found and i < n:   # if predicted entity is longer than actual entity
            if actual[i][0] == predicted[i][0]:
                tp += 1
        elif found:
            tp += 1
        else:
            pass
        
    return tp, cp
    
        
def sentence_metrics_stripped(predicted, actual):
    '''
    Return number of true positives (tp) and actual positives(cp) in between
    a predicted and target sentence.
    
    Inputs list of word labels of equal length
    '''
    
    assert len(predicted) == len(actual)
    
    cp, tp = 0, 0
    n = len(predicted)
    start_ne = list()
    for i, s in enumerate(actual):
        if s[0] == 'B':
            cp += 1
            start_ne.append(i)
    
    
    for start_i in start_ne:
        
        i = start_i
        while i < n and actual[i] in ['B','I']:
            
            if actual[i] == predicted[i]:
                found = True  
            else:
                found = False
            i += 1
            
            
        if found and i < n:   # if predicted entity is longer than actual entity
            if actual[i] == predicted[i]:
                tp += 1
        elif found:
            tp += 1
        else:
            pass
        
    return tp, cp