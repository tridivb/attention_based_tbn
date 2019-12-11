import argparse
import h5py
import os
import numpy as np
import cv2
import time
import librosa as lr
from parse import parse
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(
        description="dump epic kitchens images into hdf5 databases"
    )
    parser.add_argument(
        "--mode", choices=["rgb", "flow", "audio"], help="rgb/flow mode", default="rgb", type=str
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
    if args.mode == "rgb":
        root_path = os.path.join(root_dir, "rgb")
        out_dir = os.path.join(out_dir, "rgb")
    elif args.mode == "flow":
        root_path = os.path.join(root_dir, "flow")
        out_dir = os.path.join(out_dir, "flow")

    os.makedirs(out_dir, exist_ok=True)

    vid_path = os.path.join(root_dir, v)
    vid_id = os.path.split(v)[1]

    print("Processing {}...".format(vid_id))
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

    h5_file = os.path.join(out_dir, "{}_{}.hdf5".format(vid_id, mode))
    f = h5py.File(h5_file, "w")
    dset = f.create_dataset(vid_id, shape=(n, h, w, c), dtype=h5py.h5t.STD_U8BE, compression="gzip")
    for idx in range(n):
        img = read_image(vid_path, all_files[idx], mode)        
        dset[idx] = img
    f.close()
    print("Done. Frame data for {} saved to {}...".format(vid_id, h5_file))


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
        f = h5py.File(h5_file, "r+")
    else:
        f = h5py.File(h5_file, "w")

    try:
        sample, _ = lr.core.load(os.path.join(aud_path + "." + ext), sr=args.sr, mono=True)
    except Exception as e:
        raise Exception("Failed to read audio file {} with error {}".format(f, e))
    dset = f.create_dataset(
        vid_id, shape=sample.shape, dtype=h5py.h5t.STD_U8BE, data=sample
    )
    f.close()
    print("Done. Audio data for {} saved to {}...".format(vid_id, h5_file))


def main(args):

    out_dir = args.out_dir if args.out_dir else "./"

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
                file_format=args.file_format
            )
            for v in vid_list
        )
    else:
        Parallel(n_jobs=args.njobs)(
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
    print("Time taken: {:.3f} seconds".format(time.time() - start))


if __name__ == "__main__":
    args = parse_args()
    main(args)
