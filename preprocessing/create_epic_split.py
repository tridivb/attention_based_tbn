import os
import argparse
import random
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="create dataset split")
    parser.add_argument(
        "annotation",
        help="annoation file to read",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        help="output directory",
        dest="out_dir",
        default=os.path.dirname(os.path.realpath(__file__)),
        type=str,
    )
    parser.add_argument(
        "--mode",
        help="mode of split",
        dest="mode",
        default="random",
        choices=["random", "epic"],
        type=str,
    )
    return parser.parse_args()

def write_list_to_file(file, lst):
    with open(file, "w") as f:
        for item in sorted(lst):
            f.write("%s\n"%item)

def create_split(args):

    if args.annotation.endswith("csv"):
        df = pd.read_csv(args.annotation)
    elif args.annotation.endswith("pkl"):
        df = pd.read_pickle(args.annotation)
    else:
        raise Exception("Incorrect file extension for annotation file. Must be a csv or pkl file")

    train_list = []
    val_list = []

    for p_id in df.participant_id.unique():
        data = df.query("participant_id == @p_id")
        vid_ids = list(data.video_id.unique())
        if args.mode == "random":
            random.shuffle(vid_ids)
            train_list.extend(vid_ids[:-1])
            val_list.append(vid_ids[-1])
        elif args.mode == "epic":
            if p_id < "P25":
                train_list.extend(vid_ids)
            else:
                val_list.extend(vid_ids)

    train_list_file = os.path.join(args.out_dir, "train_split.txt")
    val_list_file = os.path.join(args.out_dir, "val_split.txt")

    write_list_to_file(train_list_file, train_list)
    write_list_to_file(val_list_file, val_list)

if __name__ == "__main__":
    
    args = parse_args()
    create_split(args)