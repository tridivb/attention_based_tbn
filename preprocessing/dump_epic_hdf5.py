import argparse
import h5py
import os
import numpy as np
import cv2
import time
import librosa as lr
from mpi4py import MPI
from parse import parse
from tqdm import tqdm
from joblib import Parallel, delayed
from utils.misc import get_time_diff


def parse_args():
    parser = argparse.ArgumentParser(
        description="dump epic kitchens images into hdf5 databases"
    )
    parser.add_argument(
        "--mode",
        choices=["rgb", "flow", "audio"],
        help="rgb/flow mode",
        default="rgb",
        type=str,
    )
    parser.add_argument(
        "--lst-file",
        help="list of videos",
        default="<path_to_epic_kitchens>/EPIC_KITCHENS_2018/vid_list_mini.csv",
        type=str,
    )
    parser.add_argument(
        "--root-dir",
        help="root dir to epic kitchens",
        default="<path_to_epic_kitchens>/EPIC_KITCHENS_2018/frames_rgb_flow",
        type=str,
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="path to save output files",
        default="./epic/hdf5",
        type=str,
    )
    parser.add_argument(
        "--ext", dest="ext", help="extension of image files", default="jpg", type=str
    )
    parser.add_argument(
        "--sr", dest="sr", help="sampling rate of audio", default=24000, type=int
    )
    parser.add_argument(
        "--file-format",
        dest="file_format",
        help="naming format of image files",
        default="frame_{:010d}.jpg",
        type=str,
    )
    parser.add_argument(
        "--flow-len",
        dest="flow_win_len",
        help="no of flow files to interleaf",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--njobs", dest="njobs", help="no of cpu cores to use", default=4, type=int,
    )
    return parser.parse_args()


def read_image(path, img_file, mode="rgb"):
    if mode == "rgb":
        img = cv2.imread(os.path.join(path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "flow":
        u_img = cv2.imread(os.path.join(path, "u", img_file), 0)
        v_img = cv2.imread(os.path.join(path, "v", img_file), 0)
        img = np.concatenate((u_img[..., None], v_img[..., None]), axis=2)
    return img


def save_images_to_hdf5(
    v,
    mode,
    root_dir,
    out_dir,
    ext="jpg",
    file_format="frame_{:010d}.jpg",
    flow_win_len=5,
):
    if mode == "rgb":
        root_dir = os.path.join(root_dir, "rgb")
        out_dir = os.path.join(out_dir, "rgb")
    elif mode == "flow":
        root_dir = os.path.join(root_dir, "flow")
        out_dir = os.path.join(out_dir, "flow")

    os.makedirs(out_dir, exist_ok=True)

    vid_path = os.path.join(root_dir, v)
    vid_id = os.path.split(v)[1]

    if mode == "rgb":
        all_files = sorted(
            filter(lambda x: x.endswith(ext), os.listdir(vid_path)),
            key=lambda x: parse(file_format, x)[0],
        )
    elif mode == "flow":

        all_files_u = sorted(
            filter(lambda x: x.endswith(ext), os.listdir(os.path.join(vid_path, "u")),),
            key=lambda x: parse(file_format, x)[0],
        )
        all_files_v = sorted(
            filter(lambda x: x.endswith(ext), os.listdir(os.path.join(vid_path, "u")),),
            key=lambda x: parse(file_format, x)[0],
        )
        if all_files_u == all_files_v:
            all_files = all_files_u
        else:
            raise Exception(
                "Count of flow files in each direction do not match for video {}".format(
                    v
                )
            )

    h, w, c = read_image(vid_path, all_files[0], mode).shape
    n = len(all_files)
    cache_size = 1024 ** 3
    chunk_size = (2500, h, w, c)

    h5_file = os.path.join(out_dir, "{}_{}.hdf5".format(vid_id, mode))
    with h5py.File(h5_file, "w", rdcc_nbytes=cache_size, driver='mpio', comm=MPI.COMM_WORLD) as f:
        dset = f.create_dataset(
            vid_id,
            shape=(n, h, w, c),
            dtype=np.uint8,
            compression="gzip",
            chunks=chunk_size,
            compression_opts=4,
        )
        for idx in range(n):
            img = read_image(vid_path, all_files[idx], mode)
            dset[idx] = img
    print("----------------------------------------------------------")
    print("Frame data for {} saved to {}.".format(vid_id, h5_file))
    print("----------------------------------------------------------")


def save_audio_to_hdf5(
    v, mode, root_dir, out_dir, ext="wav", sr=24000, file_format="P{:01d}_{:01d}.wav"
):
    root_dir = os.path.join(root_dir, "audio")
    out_dir = os.path.join(out_dir, "audio")
    os.makedirs(out_dir, exist_ok=True)

    vid_id = os.path.split(v)[1]
    aud_path = os.path.join(root_dir, vid_id)
    p_id = vid_id.split("_")[0]
    h5_file = os.path.join(out_dir, "{}_{}.hdf5".format(p_id, mode))
    if os.path.exists(h5_file):
        file_mode = "r+"
    else:
        file_mode = "w"

    try:
        sample, _ = lr.core.load(os.path.join(aud_path + "." + ext), sr=sr, mono=True)
    except Exception as e:
        raise Exception("Failed to read audio file {} with error {}".format(f, e))

    chunk_size = (sample.shape[0] // 2,)
    cache_size = chunk_size * 4

    with h5py.File(h5_file, file_mode, rdcc_nbytes=cache_size) as f:
        dset = f.create_dataset(
            vid_id,
            shape=sample.shape,
            dtype=np.float32,
            data=sample,
            compression="gzip",
            chunks=chunk_size,
            compression_opts=4,
        )
    print("Done. Audio data for {} saved to {}.".format(vid_id, h5_file))


def main(args):

    with open(args.lst_file) as f:
        vid_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    print("{} videos to process".format(len(vid_list)))
    print("----------------------------------------------------------")

    start = time.time()

    if args.mode == "audio":
        Parallel(n_jobs=1)(
            delayed(save_audio_to_hdf5)(
                v,
                args.mode,
                args.root_dir,
                args.out_dir,
                ext=args.ext,
                sr=args.sr,
                file_format=args.file_format,
            )
            for v in vid_list
        )
    else:
        results = Parallel(n_jobs=args.njobs, verbose=25)(
            delayed(save_images_to_hdf5)(
                v,
                args.mode,
                args.root_dir,
                args.out_dir,
                ext=args.ext,
                file_format=args.file_format,
                flow_win_len=args.flow_win_len,
            )
            for v in vid_list
        )

    print("Done")
    print("----------------------------------------------------------")
    print("Time taken[HH:MM:SS]: {}".format(get_time_diff(start, time.time())))


if __name__ == "__main__":
    args = parse_args()
    main(args)
