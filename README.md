# keras-NER
Named entity recognition with keras. 

This repository contains an implementation of named entity recognition architectures from ["Named Entity Recognition with Bidirectional LSTM-CNNs"](https://aclanthology.org/Q16-1026.pdf).
The network features a word-level bidirectional LSTM along with character-level CNNs. 

This repository contains code for converting sentences and words into index tensors for use with Keras' embeddings layer in addition to creating batching and shuffling 
tools and Keras generators. Training and validation code is also included. 


```
.
├── data                      # contains the CoNLL 2003 named entity recognition data set
├── models                    # keras models
├── batch_utils.py            # functions for creating mini-batches based on sentence length
├── feature_utils.py          # functions for encoding words and characters as index vectors and tensors
├── generators.py             # Keras generators for use with the models contained in /models
├── process_conll2003.py      # functions for processing the CoNLL 2003 data
├── train_nn.py               # train the model
├── validation.py             # validate the model
└── README.md
```

Better documentation and code structure on its way. 
