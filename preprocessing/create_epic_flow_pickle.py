import argparse
import os
import numpy as np
import cv2
import time
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(
        description="dump epic kitchens flow images into pickle files"
    )
    parser.add_argument(
        "annotation_file",
        help="list of annotations",
        default="<path_to_epic_kitchens>/annotations/EPIC_train_action_labels.csv",
        type=str,
    )
    parser.add_argument(
        "root_dir",
        help="root dir to epic kitchens",
        default="<path_to_epic_kitchens>/EPIC_KITCHENS_2018/frames_rgb_flow/flow/<train or test>",
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
    assert os.path.exists(
        os.path.join(path, "u", img_file)
    ), "{} file does not exist".format(os.path.join(path, "u", img_file))
    u_img = cv2.imread(os.path.join(path, "u", img_file), 0)
    assert os.path.exists(
        os.path.join(path, "u", img_file)
    ), "{} file does not exist".format(os.path.join(path, "v", img_file))
    v_img = cv2.imread(os.path.join(path, "v", img_file), 0)
    img = np.concatenate((u_img[..., None], v_img[..., None]), axis=2).astype(np.uint8)
    return img


def integrity_check(file):
    try:
        with np.load(file) as data:
            _ = data["flow"]
            data.close()
            return True
    except:
        print("{} is corrupted. Overwriting file.".format(file))
        return False


def save_images_to_pickle(
    record,
    root_dir,
    out_dir,
    win_len,
    ext="jpg",
    file_format="frame_{:010d}.jpg",
    attempts=10,
):

    vid_id = record["video_id"]
    vid_path = os.path.join(root_dir, record["participant_id"], vid_id)

    out_dir = os.path.join(out_dir, "flow_pickle", vid_id)
    os.makedirs(out_dir, exist_ok=True)

    start_frame = max(record["start_frame"] // 2, 1)
    end_frame = max(record["stop_frame"] // 2, 2)

    full_read = True
    for idx in range(start_frame, end_frame + 1 - win_len):
        out_file = os.path.join(
            out_dir, os.path.splitext(file_format.format(idx - 1))[0] + ".npz"
        )
        if os.path.exists(out_file) and integrity_check(out_file):
            full_read = True
            continue
        else:
            for a in range(attempts):
                if full_read:
                    img = []
                    for i in range(win_len):
                        img.append(read_image(vid_path, file_format.format(idx + i)))
                else:
                    img = [img[:, :, 2:]]
                    img.append(read_image(vid_path, file_format.format(idx + win_len)))
                img = np.concatenate(img, axis=2)
                np.savez_compressed(out_file, flow=img)
                if integrity_check(out_file):
                    full_read = False
                    break
                elif a == attempts - 1:
                    print(
                        "Unable to save {} properly. File might be corrupted".format(
                            out_file
                        )
                    )


def main(args):

    annotations = pd.read_csv(args.annotation_file)

    start = time.time()

    print(
        "Processing {} video annoations with {} concurrent workers".format(
            annotations.shape[0], args.njobs
        )
    )
    print("----------------------------------------------------------")
    # for rid, r in annotations.iterrows():
    #     save_images_to_pickle(
    #         r,
    #         args.root_dir,
    #         args.out_dir,
    #         args.win_len,
    #         ext=args.ext,
    #         file_format=args.file_format,
    #     )
    results = Parallel(n_jobs=args.njobs, verbose=5)(
        delayed(save_images_to_pickle)(
            r,
            args.root_dir,
            args.out_dir,
            args.win_len,
            ext=args.ext,
            file_format=args.file_format,
        )
        for _, r in annotations.iterrows()
    )

    print("Done")
    print("----------------------------------------------------------")
    print("Time taken[HH:MM:SS]: {}".format(get_time_diff(start, time.time())))


if __name__ == "__main__":
    args = parse_args()
    main(args)
