import argparse
import os
import librosa as lr
import numpy as np
from tqdm import tqdm


def parse_args():
    """
    Helper function to parse command line arguments
    """

    parser = argparse.ArgumentParser(description="dump audio samples into binary files")
    parser.add_argument("audio_dir", help="path to audio files", type=str)
    parser.add_argument(
        "--sr", dest="sr", help="sampling rate of audio", default=24000, type=int
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="path to save output files",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "audio_pickle"
        ),
        type=str,
    )
    parser.add_argument(
        "--ext", dest="ext", help="extension of audio files", default="wav", type=str
    )
    return parser.parse_args()


def main(args):
    assert os.path.exists(args.audio_dir), "Audio path {} does not exist".format(
        args.audio_dir
    )

    os.makedirs(args.out_dir, exist_ok=True)

    rejected = []
    print("Processing audio files ...")
    # Iterate over the directory and save the audio for each video into a separate numpy file
    for _, _, files in os.walk(args.audio_dir):
        for f in tqdm(files):
            if f.endswith(args.ext):
                try:
                    sample, _ = lr.core.load(
                        os.path.join(args.audio_dir, f), sr=args.sr, mono=True
                    )
                except Exception as e:
                    print("Failed to read audio file {} with error {}".format(f, e))
                    rejected.append(f)
                npy_file = os.path.splitext(f)[0] + ".npy"
                np.save(
                    os.path.join(args.out_dir, npy_file),
                    sample,
                    allow_pickle=True,
                    fix_imports=True,
                )

    print("Finished creating numpy files.")
    print("----------------------------------------------------------")

    if len(rejected) > 0:
        print("List of rejected files: {}".format(rejected))


if __name__ == "__main__":
    args = parse_args()
    main(args)
