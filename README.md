# netam
Neural networks to model BCR affinity maturation

## TODOs?

* try adding annotation to the sequences and training on it
* think about boundary cases of beginning and end of sequence
* rerun everything with 500 long bases
* rename `CNNPP` model to `CNPE` model

## Goals

* modern techniques such as regularization and transformers
* a comprehensive survey, and provide useful software with rigorous metrics
* comparison to oracle


## Results

* `cnn.ipynb`: Hyperparameter optimization for CNN model
* `cnn_1mer.ipynb`: Hyperparameter optimization for CNN1Mer model
* `cnnmlp.ipynb`: Adding a hidden layer in the final layer is bad
* `cnnpp.ipynb`: Adding a positional encoding to the CNN
* `cnnxformer.ipynb`: Adding a transformer to the CNN makes it worse
* `data-description.ipynb`: Exploration of SHMoof data sets
* `fivemer.ipynb`: L2 regularizing the 5mer model doesn't help
* `model_comparison.ipynb`: Main model comparison notebook
* `noof.ipynb`: A transformer on the kmer embeddings is not a good model
* `penalize-site-rates.ipynb`: Trying to penalize the site rates of SHMoof
* `persite_wrapper.ipynb`: Developing the `PersiteWrapper` and showing that regularizing it doesn't help
* `reshmoof.ipynb`: Re-fitting the SHMoof model, playing with regularization, showing that per-site mutability tracks per-site motif mutability
* `twolength.ipynb`: An experiment trying to see if stratifying the SHMoof model into long and short components doesn't help
 

## Conclusions
* CNN using kmer embeddings work, and can be parameter-sparse
* transformers aren't good for this problem, and positional encoding appears to hurt