# attention_based_tbn
Attention based Temporal Binding Network\
This work is part of my Master Thesis "A Study of Audio effectiveness for Action Recognition from egocentric videos".\

The rest of the Readme, Installation instructions with further code documentation is coming soon.
\
## Getting Started

Clone the repo and set it up in your local drive.

```
git clone https://github.com/tridivb/epic_fusion_feature_extractor.git
```

### Prerequisites

Python >= 3.7\
[Pytorch](https://pytorch.org/)  >= 1.4\
Hydra-core == 0.11
[OmegaConf](https://github.com/omry/omegaconf) == 1.4.1\
[Numpy](https://numpy.org/) \
[Tqdm](https://github.com/tqdm/tqdm) \
[PIL](https://pillow.readthedocs.io/en/stable/) \
[Parse](https://pypi.org/project/parse/) \
[librosa](https://librosa.github.io/librosa/) \
\
The requirement.txt file can be used to install the dependencies.
```
pip install -r requirements.txt
```


### Setting up the data
\

```
|---<path to dataset>
|   | rgb_prefix
|   |   |--- video_1
|   |   |   |--- img_0000000000
|   |   |   |--- x_0000000000
|   |   |   |--- y_0000000000
|   |   |   |--- .
|   |   |   |--- .
|   |   |   |--- .
|   |   |--- .
|   |   |--- .
|   |   |--- .
|   |   |--- video_100
|   |   |   |--- img_0000000000
|   |   |   |--- x_0000000000
|   |   |   |--- y_0000000000
|   |   |   |--- .
|   |   |   |--- .
|   |   |   |--- .
|   |   |--- .
|   |   |--- .
|   |   |--- .
|   audio_prefix
|   |   |--- video_1.wav
|   |   |--- .
|   |   |--- .
|   |   |--- video_100.wav
|   |   |--- .
|   |   |--- .
|   |   |--- .
|   |---vid_list.txt
```

### Installing
\
TODO [INSTALL](install/INSTALL.md)
```

### Configure the paramters

TODO

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.\
\
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
