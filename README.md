# beyond-part-models-gluon
Implementation of <Beyond Part Models: Person Retrieval with Refined Part Pooling>, using gluon(mxnet)

## Codes Instruction 
I will introduce my code in this part.<br>
The `experiment` contains all scripts used for experiments.

### data
In the `data` package, I put the scripts about *loading data* and *save snapshots*.<br>
    `textdataset.py` defines the TextDataset, which define the dataset defined by a text file.
    `transform.py` defines the classes Transformer. 
        `ListTransformer` is used for training.
        `Market1501_Transformer` is used for market_1501 and duke testing.
    `saver.py` defines a object for saving snapshots, only when the result is better.

### metric
In the `metric` package, I put the `reidmetric.py`, which define a metric designed for Re-ID.

### model
The `model` package contains the design of the model structures.<br>
The `PCBRPPNet` is the implementation of the **beyond part model**. <br>
Params `withpcb`, `withrpp` and `feature_shared_weight` are designed for the different situation.

### process

