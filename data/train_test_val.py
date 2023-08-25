'''
This script helps in converting the folder containing images arranged for image classification into train, test and validate.
'''

import os
import math
import shutil
import random
from argparse import ArgumentParser
import logging

def main():
    src_root_path = args.src
    dest_root_path = args.dest
    
    src_sub_dirs = os.listdir(src_root_path)

    shutil.rmtree(dest_root_path,ignore_errors=True)
    os.makedirs(dest_root_path, exist_ok=True)
    logging.info(f"cleaned and created the destination folder \n {dest_root_path}")

    dest_sub_dir = ["train", "test", "val"]

    # for dir in dest_sub_dir:
    #     os.makedirs(os.path.join(dest_root_path, dir))

    for dir in src_sub_dirs:
        sub_list = os.listdir(os.path.join(src_root_path, dir))
        random.shuffle(sub_list)
        total_len = len(sub_list)

        length = []
        jag_arr = []

        for i in range(3):
            length.append(math.floor(args.split[i]*total_len))
    
        jag_arr.append(sub_list[0:length[0]])
        jag_arr.append(sub_list[length[0]:length[0]+length[1]])
        jag_arr.append(sub_list[length[0]+length[1]: total_len])

        for i, arr in enumerate(jag_arr):
            dest_root_1 = os.path.join(dest_root_path, dest_sub_dir[i])
            dest_root_2 = os.path.join(dest_root_1, dir)

            src_root_1 = os.path.join(src_root_path, dir)

            os.makedirs(dest_root_2)

            for img_name in arr:
                try:
                    shutil.copy(os.path.join(src_root_1, img_name), dest_root_2)

                except Exception as e:
                    logging.error(e)
        logging.info(f"splitting done for {dir} folder")


def get_args():
    parser = ArgumentParser()

    parser.add_argument("--src",default=r"C:\Users\ATI-G2\Documents\python\ECG\data\vit-data", type=str)
    parser.add_argument("--split", default=(0.95, 0.02, 0.03), type=tuple)
    parser.add_argument("--dest",default=r"C:\Users\ATI-G2\Documents\python\ECG\data\dataRoot", type=str)

    return parser.parse_args()


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    main()