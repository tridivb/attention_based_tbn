import argparse
import os
import numpy as np
import cv2
import time
from parse import parse
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(
        description="dump epic kitchens flow images into pickle files"
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
        default="<path_to_epic_kitchens>/EPIC_KITCHENS_2018/frames_rgb_flow/flow",
        type=str,
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="path to save output files",
        default="./epic/flow",
        type=str,
    )
    parser.add_argument(
        "--ext", dest="ext", help="extension of image files", default="jpg", type=str
    )
    parser.add_argument(
        "--file-format",
        dest="file_format",
        help="naming format of image files",
        default="frame_{:010d}.jpg",
        type=str,
    )
    parser.add_argument(
        "--win-len",
        dest="win_len",
        help="number of flow frames to read",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--njobs", dest="njobs", help="no of cpu cores to use", default=4, type=int,
    )
    return parser.parse_args()


def get_time_diff(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = round((end_time - start_time) % 60)
    return (hours, minutes, seconds)


def read_image(path, img_file):
    u_img = cv2.imread(os.path.join(path, "u", img_file), 0)
    v_img = cv2.imread(os.path.join(path, "v", img_file), 0)
    img = np.concatenate((u_img[..., None], v_img[..., None]), axis=2).astype(np.uint8)
    return img


def save_images_to_npy(
    v,
    root_dir,
    out_dir,
    win_len,
    ext="jpg",
    file_format="frame_{:010d}.jpg",
    flow_win_len=5,
):

    vid_path = os.path.join(root_dir, v)
    vid_id = os.path.split(v)[1]

    out_dir = os.path.join(out_dir, "flow_pickle", vid_id)
    os.makedirs(out_dir, exist_ok=True)

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
            "Count of flow files in each direction do not match for video {}".format(v)
        )

    h, w, c = read_image(vid_path, all_files[0]).shape
    n = len(all_files)

    for idx in range(n - win_len):
        if idx == 0:
            img = []
            for i in range(win_len):
                img.append(read_image(vid_path, all_files[idx + i]))
        else:
            img = [img[:, :, 2:]]
            img.append(read_image(vid_path, all_files[idx + win_len]))
        img = np.concatenate(img, axis=2)
        out_file = os.path.join(out_dir, os.path.splitext(all_files[idx])[0])
        # np.save(out_file, img)
        np.savez_compressed(out_file, arr=img)

    print("----------------------------------------------------------")
    print("Flow pickles for {} saved to {}.".format(vid_id, out_dir))
    print("----------------------------------------------------------")


def main(args):

    with open(args.lst_file) as f:
        vid_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    start = time.time()

    print("Processing {} videos with {} concurrent workers".format(len(vid_list), args.njobs))
    print("----------------------------------------------------------")
    results = Parallel(n_jobs=args.njobs)(
        delayed(save_images_to_npy)(
            v,
            args.root_dir,
            args.out_dir,
            args.win_len,
            ext=args.ext,
            file_format=args.file_format,
        )
        for v in vid_list
    )

    print("Done")
    print("----------------------------------------------------------")
    print("Time taken[HH:MM:SS]: {}".format(get_time_diff(start, time.time())))


if __name__ == "__main__":
    args = parse_args()
    main(args)
