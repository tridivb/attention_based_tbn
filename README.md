# Attention based Temporal Binding Network

This work is part of my Master Thesis titled "A Study of Audio effectiveness for Action Recognition from egocentric videos". The complete writeup can be found at [Master Thesis](thesis/Master_Thesis.pdf).

## Getting Started
Clone the repo and set it up in your local drive

```
git clone https://github.com/tridivb/attention_based_tbn.git
```

## Prerequisites
1. Ubuntu 16x/18x (The framework has not been tested beyond these two systems.)
2. Cuda 10.2
3. Miniconda/Anaconda
4. Install the required python environment via the instructions in [INSTALL.md](install/INSTALL.md)

## Preprocessing
Setup the dataset in the below structure:

```
├── root
|   ├── video1
|   |   ├── img_0000000000.jpg
|   |   ├── x_0000000000.jpg
|   |   ├── y_0000000000.jpg
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000100.jpg
|   |   ├── x_0000000100.jpg
|   |   ├── y_0000000100.jpg
|   ├── .
|   ├── .
|   ├── .
|   ├── video10000
|   |   ├── img_0000000000.jpg
|   |   ├── x_0000000000.jpg
|   |   ├── y_0000000000.jpg
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000250.jpg
|   |   ├── x_0000000250.jpg
|   |   ├── y_0000000250.jpg
|   ├── audio
|   |   ├── video1.wav
|   |   ├── video2.wav
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── video10000.wav
|   ├── annotations
|   |   ├── annotation_file1.csv
|   |   ├── annotation_file2.csv
|   |   ├── .
|   |   ├── .
|   |   ├── .

```

Since we primarly work with the [Epic-Kitchens 55](https://github.com/epic-kitchens), scripts are provided in [preprocessing](preprocessing) to automate this step on the Epic-Kitchens-55 dataset. Scripts to create symlinks from the visual frames and extract the audio from videos are included inside the directory. One can also setup their own dataset in the above format to use with the framework.

The dataset splits are done by providing the video names in separate train/validation files in the [data](data) directory. There are two splits for train and validation as "Seen" and "Unseen". The "Seen" set contains 14 videos held out for validation and the "Unseen" set contains all videos with person id "P_25" and above for validation.

You can randomly generate your own "Seen" set with the [create_epic_split.py](preprocessing/create_epic_split.py) script. Move those files to the [data](data) directory once they are generated.

## Pretrained Weights
Download the pretrained weights for imagenet and kinetics as:

```
cd weights
./download.sh
```

Some of the trained models for our baselines and experiments can be found at this [link](https://drive.google.com/drive/folders/19rbjMRqtOdLv_b1WnGTq_BJ2TsMwDFsO?usp=sharing).

## Configuration
The list of configuration parameters can be found at [CONFIG.md](config/CONFIG.md).

## Training

TODO

## Testing

TODO

## Results

TODO

## Epic-Kitchens Evaluation

TODO

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please note, the original EPIC-Fusion framework is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Please respect the original licenses as well.

## Acknowledgments

1. EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition
    ```
    @InProceedings{kazakos2019TBN,
    author    = {Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima},
    title     = {EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2019}
    }
    ```

2. Readme Template -> https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
