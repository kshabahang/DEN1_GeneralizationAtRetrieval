# DEN1_GeneralizationAtRetrieval
Generalization at retrieval with a Dynamic Eigen Net


###RUNNING THE TOY EXAMPLES###

Run toy_example.py to produce data presented in the first three tables. You can change between the linear, saturate (BSB), and stp (DEN) versions of the associative net by changing the value of the hyperparameter, "feedback".


hyperparameters include:
   eps         - convergence criterion
   eta         - learning rate (kept at 1)
   alpha       - stp scaling weight (kept at 1)
   V0          - the maximum size of the vocabulary
   N           - the number of features used to represent vocab items (should be same as V0 for localist rep.)
   localist    - same as one-hot encoding
   distributed - random ternary vectors
   explicit    - explicitly represent the weight matrix (i.e., for doing matrix-multiplication)
   idx_predict - the slot whose most active entrie we want the net to generate
   numBanks    - number of banks 
   numSlots    - number of slots (these are a special case of banks, where only a single item is in each bank)
       These two can be left the same for the sake of the current manuscript's simulations.
   C           - this is the saturation constant used for BSB (i.e., when the "feedback" is set to "saturate"
   mode        - this is to switch between numpy, cuda, and sparse formats
   feedback    - this is the type of update-function we use to drive retrieval (recurrence)
   gpu         - set to true if you have cuda


###RUNNING THE BIGRAM EXAMPLES####
The first step is to load the corpus, extract a vocabulary of all the unique words. The matrix E only applies when using distributed representations. You can ignore it when using a localist representation. I note here that the reason this stage is done in multiple stages is to facilitate counting multiple separate chunks of the corpus in parallel. Only one chunk should suffice for corpuses that are not too much larger than 100 mbs. It is assumed that the corpus is a subdirectory inside rsc, and that the corpus txt file is the has the exact same name (not including the file extension).

Initialize the corpus by running, "count.py init <corpus_path>", where <corpus_path> may be "TASA" if "TASA" is a directory inside the rsc folder, and it contains a text file named "TASA.txt". It's assumed that the corpus contains sentences separated by newline (\n) characters.

Once you got your corpus processed, run "bigram_gen.py TASA" to run model on the bigrams. This stage can take a good deal of time, so you likely want to set the job up and get on with other business. Change the value of the variable "sweep_idx" to run on different bigrams sets.



If this work was useful to you and you want to somehow thank my work, donations are welcome:
<p>BTC: bc1q3dtvmf0gd7gqmcwlv7kwfkd6wj3023f8pe3lgl</p>
<p>ETH: 0x5831aa28D2378Ae5333f57B3C2d8FeC3C736eEeb</p>
<p>XMR: 44q99xTChW3B8dNykAGRza66TRZi2wpnAZtj2FuGwwL9H8shiXJYwgcicGf529uufyRDBMsLTLXAcKWohQRHvvdfUw4fWm2</p>
<p>DODGE: DEhsBqavQmY2i7RgZQCsjXeTY9kceuy454</p>
<p>LTC: ltc1qq9gdv7tpmwutdxvap05t049rvm96qtmmmtshs2</p>



 
