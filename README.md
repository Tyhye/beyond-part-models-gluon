# beyond-part-models-gluon
Implementation of &lt;Beyond Part Models: Person Retrieval with Refined Part Pooling&gt;, using gluon(mxnet)

## Codes Instruction 
I will introduce my code in this part.
The `experiment` contains all scripts used for experiments.

### data
In the `data` package, I put the scripts about *loading data* and *save snapshots*.
+ `textdataset.py` defines the TextDataset, which define the dataset defined by a text file.
+ `transform.py` defines the classes Transformer. 
    - `ListTransformer` is used for training.
    - `Market1501_Transformer` is used for market_1501 and duke testing.
+ `saver.py` defines a object for saving snapshots, only when the result is better.

### metric
In the `reidmetric.py`, which define a metric class designed for Re-ID.

### model
The `model` package contains the design of the model structures.
The `PCBRPPNet` is the implementation of the **beyond part model**.
Params `withpcb`, `withrpp` and `feature_shared_weight` are designed for the different situation.

### process
I defined two kind of processor to control the processes of the training and testing.
We could implement function for our own experiments.
+ `epochprocessor.py` is designed for using *epoch* to control the number of training times 
+ `iterprocessor.py` is designed for using *iter* to control the number of training times 
