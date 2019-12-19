import argparse
import os
import numpy as np
import cv2
import time
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_args():
    """
    Helper function to parse command line arguments
    """
    
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
    """
    Helper function to calculate time difference

    Args
    ----------
    start_time: float
        Start time in seconds since January 1, 1970, 00:00:00 (UTC)
    end_time: float
        End time in seconds since January 1, 1970, 00:00:00 (UTC)

    Returns
    ----------
    hours: int
        Difference of hours between start and end time
    minutes: int
        Difference of minutes between start and end time
    seconds: int
        Difference of seconds between start and end time
    """

    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = round((end_time - start_time) % 60)
    return (hours, minutes, seconds)


def read_image(path, img_file):
    """
    Helper function to read image file(s)

    Args
    ----------
    path: str
        Source path of image file
    img_file: str
        Name of image file

    Returns
    ----------
    img: np.ndarray
        A numpy array of the image
    """

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
    """
    Helper function to check integrity of a compressed numpy file

    Args
    ----------
    file: str
        Absolute location of compressed numpy file

    Returns
    ----------
    check_flag: bool
        A flag confirming if the file is ok or not
    """

    check_flag = None
    try:
        with np.load(file) as data:
            _ = data["flow"]
            data.close()
        check_flag = True
    except:
        print("{} is corrupted. Overwriting file.".format(file))
        check_flag = False
    return check_flag

def save_images_to_pickle(
    record,
    root_dir,
    out_dir,
    win_len,
    ext="jpg",
    file_format="frame_{:010d}.jpg",
    attempts=10,
):
    """
    Helper function to iterate over each frame of a trimmed action segment and
    save the array of stacked flow frames to a compressed numpy file

    Args
    ----------
    record: pd.dataframe row
        Row with trimmed action segmentation of untrimmed video
    root_dir: str
        Root directory of dataset
    out_dir: str
        Directory to store output files
    win_len: int
        No of optical flow frames to stack
    ext: str, default="jpg"
        Extension of optical flow files
    file_format: str, default="frame_{:010d}.jpg"
        File naming format
    attempts: int, default=10
        No of attempts to write files

    """

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
        # If file exists and is ok, skip
        if os.path.exists(out_file) and integrity_check(out_file):
            full_read = True
            continue
        else:
            for a in range(attempts):
                # Create the whole flow stack for non-sequential frame indices
                if full_read:
                    img = []
                    for i in range(win_len):
                        img.append(read_image(vid_path, file_format.format(idx + i)))
                # Only append the data from new flow files in case of sequential frame indices
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
    # Trimmed action segments to be processed in parallel
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
