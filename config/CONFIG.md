# Configuration

The configuration module has been built with [Hydra 0.11](https://hydra.cc/docs/intro/). The configuration groups have been divided as follows:

```
|   main
|   |---- data
|   |---- hydra
|   |---- model
|   |---- test
|   |---- train
|   |---- val

```

You can customize the configurations as per your model requirements or create separate files inside each group. Please check the hydra documentation on how to use config groups.

## Main module

The main module is initialized by [config.yaml](config.yaml). A complete list of the configuration parameters available with it are listed below:

```
# Define the files for default config groups here
defaults:
  - data: tbn_data
  - model: tbn_model
  - train: tbn_train
  - val: tbn_val
  - test: tbn_test
  - hydra: custom
# Number of pytorch workers
num_workers: 8
# (Optional) Specific Gpu ids to use, if not specified the framework will use all available Gpus
gpu_ids: []
# Name of experiment
exp_name: "attention_test/"
# Dataset root directory
data_dir: "/media/data/tridiv/epic"
# Output directory
out_dir: "/media/data/tridiv/epic"
```

#### Data Config


```
data:
  # Name of dataset
  dataset: "epic"
  # type of sampling. Choose between "sync" and "async"
  sampling: "sync"
  rgb:
    # flag to enable rgb modality
    enable: True
    # directory prefix for rgb files present under dataset root
    dir_prefix: "links"
    # rgb file extension
    file_ext: "jpg"
    # rgb channel mean
    mean: [0.408, 0.459, 0.502]
    # rgb channel standard deviation
    std: [1.0, 1.0, 1.0]
  flow:
    # flag to enable optical flow modality
    enable: False
    # flag to enable reading flow from pickled files
    read_flow_pickle: False
    # directory prefix for flow files present under dataset root
    dir_prefix: "flow_pickle"
    # flow file extension
    file_ext: "jpg"
    # number of flow files to interleave. The total number be frames will be 2 x win_length as each frame is represented by two files.
    win_length: 5
    # flow channel mean
    mean: [0.502]
    # flow channel standard deviation
    std: [1.0]
  audio:
    # flag to enable audio modality
    enable: True
    # flag to enable reading audio from pickled files
    read_audio_pickle: False
    # directory prefix for audio files present under dataset root
    dir_prefix: "audio"
    # audio sampling rate in Hz
    sampling_rate: 24000
    # length of audio in seconds
    audio_length: 2.1
    # type of spectrogram. choose between "stft" (Short-time Fourier Transfor) and "logms" (log-mel)
    spec_type: "stft"
    # audio file extension
    file_ext: "wav"
    # audio dropout probability (only usable with multi-modal networks)
    dropout: 0
  # video input fps
  vid_fps: 60
  # scale size of input visual frame during training
  train_scale_size: 256
  # crops size of input visual frame during training
  train_crop_size: 224
  # scale size of input visual frame during testing
  test_scale_size: 256
  # crops size of input visual frame during testing
  test_crop_size: 224
  # set manual seed for reproducability
  manual_seed: 0
```

#### Model Config

```
model:
  # model architecture. Choose between "bninception", "vgg" and "resnet"
  arch: "bninception"
  attention:
    # flag to turn on/off attention layer
    enable: True
    # flag for positional encoding layer
    use_pe: true
    # type of attention. choose between "mha" (multi-headed), "unimodal" and "proto" (prototype)
    type: "mha"
    # flag to turn on/off using gumble-softmax during training
    use_gumbel: True
    # flag to use fixed attention instead of learnable ones
    use_fixed: False
    # type of prior for multi-headed attention prior loss
    prior_type: "gaussian"
    # number of attention head in multi-headed attention
    attn_heads: 4
    # multi-headed attention dropout probability
    attn_dropout: 0.5
    # flag to turn on/off prior loss
    use_prior: False
    # type of prior loss
    wt_loss: "kl"
    # prior loss decay factor
    wt_decay: 0.25
    # prior loss reduction type
    loss_reduction: "batchmean"
    # flag to turn on/off contrast loss
    use_contrast: False
    # contrast loss threshold for binary masking
    contrast_thresh: 0.1
    # contrast loss decay factor
    contrast_decay: 0.25
    # flag to turn on/off entropy loss
    use_entropy: False
    # entropy loss decay factor
    entropy_decay: 0.25
    # entropy loss threshold value
    entropy_thresh: 0.2
    # step size for decay factor in epochs
    decay_step: 10
  resnet:
    # resnet model depth. choose between 18, 34, 50, 101 and 152
    depth: 101
  vgg:
    # vgg model type. choose between 11, 11bn, 16 and 16bn
    type: "16"
  # flag to freeze base model
  freeze_base: True
  # type of freezing. choose between "all" and "partialbn" 
  freeze_mode: "partialbn"
  # number of classes
  num_classes: { class_name_1: number_1, class_name_2: number_22 }
  # temporal aggregation type. Note: Only "avg" is usable for now.
  agg_type: "avg"
  # dropout probability in fusion layer
  fusion_dropout: 0.5
  # classification loss type
  loss_fn: "crossentropy"
  # output directory to save checkpoints
  checkpoint_dir: "<checkpoint_dir>"
```

#### Train Config

```
train:
  # flag to turn on/off training
  enable: True
  # annotation file
  annotation_file: "annotations/<annotation_file>.csv"
  # list of videos to use in training
  vid_list: "data/<train_split_name>.txt"
  # train batch size
  batch_size: 12
  # total number of epochs to train the model
  epochs: 30
  optim:
    # optimizer type. choose between "sgd" and "adam"
    type: "sgd"
    # initial learning rate
    lr: 1e-2
    # momentum for sgd
    momentum: 0.9
    # weight decay factory
    weight_decay: 0
    # gradient accumulator step in iterations
    accumulator_step: 1
  scheduler:
    # learning rate scheduler step
    lr_steps: [20]
    # learning rate decay factor
    lr_decay: 1e-1
  warmup:
    # flag to turn on/off learning rate warmup
    enable: False
    # learning rate multiplier during warmup
    multiplier: 1
    # number of epochs for learning rate warmup
    epochs: 5
  # gradient clipping factor
  clip_grad: 20
  # number of temporal segments during training
  num_segments: 3
  # checkpoint file to load before resuming training
  pre_trained: ""

```

#### Val Config

```
val:
  # flag to turn on/off validation
  enable: True
  # list of videos to use in training
  vid_list: "data/val_split_seen.txt"
  # validation batch size
  batch_size: 2
  # top-k accuracy metrics to use for validation
  topk: [1, 5]
  # number of temporal segments during validation
  num_segments: 25
```

#### Test Config

```
test:
  # flag to turn on/off testing
  enable: False
  # annotations files to use for testing. multiple annotation files can be provided for processing. each file will be processed once.
  annotation_file: ["annotations/<annotation_file_1>.csv", "annotations/<annotation_file_2>.csv"]
  # list of videos to process. Leave this blank if you want to process all the annotations. In case of filtering from multiple annotation files, put list of all videos in a single file.
  vid_list: ""
  # testing batch size
  batch_size: 2
  # top-k accuracy metrics to use for testing
  topk: [1, 5]
  # number of temporal segments during testing
  num_segments: 25
  # flag to turn on/off saving results in json files. This is meant for use to generate files for the epic-kitchens test server submissions.
  save_results: False
  # name of results files. Number of results files must be equal to the number of annotation files.
  results_file: ["seen.json", "unseen.json"]
  # pre_trained weights to load for evaluation
  pre_trained: ""

```


## Visualization module

The visualization module is initialized by the [config_vis.yaml](config_vis.yaml).It is meant to visualize results from trained models and hence the same configuration parameters as the main module except the train and val parameters, which are not needed. The model configurations should be carefully selected so that it is the same as that used while training the model. A simple way to do this would be to copy the config.yaml from the log directory. The configs used for training a model can be found at `<log_root>/<experiment_name>/<run number>/.hydra/config.yaml`. Copy the configs from this file, remove the train and val sections and run it with the visualization module.