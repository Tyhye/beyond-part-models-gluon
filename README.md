# beyond-part-models-gluon
Implementation of &lt;Beyond Part Models: Person Retrieval with Refined Part Pooling&gt;, using gluon(mxnet)

## Memo
* [x] Model Implementation
* [x] Metric Coding.
* [x] Data Loading.
* [x] Result Saving.
* [x] Process Control.
* [ ] Train Process.
* [ ] Show Result.

## Result

## Usage
```
Usage: 
    python main.py [options]
    python main.py --withpcb [options]
    python main.py --withpcb --withrpp [options]

General Options:
    -h, --help                  Print this message
    --logfile=<str>             File path for saving log message. 
    --device_type=<str>         Device Type for running the model [default: cpu]
    --device_id=<int>           Device ID for running the model [default: 0]
    
Network Options:
    --basenet_type=<str>        BaseNet type for Model [default: resnet50]
    --classes_num=<int>         Output classes number of the network [default: 751]
    --feature_channels=<int>    Feature channels of the network [default: 512]
    --partnum=<int>             The number of the pcb parts. [default: 6]
    --feature_weight_share      If the six partnum share weights.
    --base_not_pretrain         If the base network don't pretrained on ImageNet
    --pretrain_path=<str>       Path to pretrained model. 

Training Setting Options:
    --Optim=<str>               Optimizer Type [default: sgd]
    --LRpolicy=<str>            Learning rate policy [default: multistep]
    --Stones=<str>              Step stone for multistep policy [default: [40,]]

    --max_epochs=<int>          Max Train epochs [default: 60]
    --log_epochs=<int>          Log step stone [default: 5]
    --snap_epochs=<int>         Snap step stone [default: 5]
    --Snap=<str>                Model state dict file path [default: saved/]

Data Options:
    --resize_size=<tuple>       Image resize size tuple (height, width) [default: (384, 128)]
    --crop_size=<tuple>         Image crop size tuple (height, width) [default: (384, 128)]
    --batchsize=<int>           Batchsize [default: 8]

Train Data Options:
    --trainList=<str>           Train files list txt [default: datas/Market1501/train.txt]
    --trainIMpath=<str>         Train sketch images path prefix [default: datas/img_gt/]
    
Test Data Options:
    --queryList=<str>           Query files list txt [default: datas/Market1501/query.txt]
    --queryIMpath=<str>         Query sketch images path prefix [default: datas/Market1501/]
    --galleryList=<str>         Gallery files list txt [default: datas/Market1501/gallery.txt]
    --galleryIMpath=<str>       Gallery sketch images path prefix [default: datas/Market1501/]
    
Learning Rate Options:
    --learning_rate=<float>     Learning rate for training process [default: 0.01]
    --weight_decay=<float>      Weight decay for training process [default: 0.0005]
    
    --base_not_train            If don't train base network.
    --base_lr_scale=<float>     Learing rate scale rate for the base network [default: 0.1]
    
    --tail_not_train            If don't train tail module, when w/o pcb and w/o rpp.
    --tail_lr_scale=<float>     Learing rate scale rate for the tail module.
    
    --pcb_not_train             If the pcb module are not trained.
    --pcb_lr_scale=<float>      Learing rate scale rate for the pcb module.
    
    --rpp_not_train             If don't train the rpp module.
    --rpp_lr_scale=<float>      Learing rate scale rate for the rpp module.
```

## Note
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
