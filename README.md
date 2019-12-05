# attention_based_tbn
Attention based Temporal Binding Network\
\
To run, set the following directories in the config/config.yaml
```
DATA:
  DATA_DIR: "/path/to/dataset/root"
  ANNOTATION_FILE: "path/to/EPIC_train_action_labels.csv"
  FRAME_DIR_PREFIX: "path/to/frame/dir"
  AUDIO_DIR_PREFIX: "path/to/audios/dir"
MODEL:
  CHECKPOINT_DIR: "path/to/save/checkpoints"
LOG_DIR: "path/to/log/dir"
```

Execute the following command:
```
python main.py ./config/config.yaml
```

