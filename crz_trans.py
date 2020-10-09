import argparse
import os
import pprint

def get_args():
    parser = argparse.ArgumentParser(description="Using model_pencil2 to transform images.")
    parser.add_argument("-i", "--inpath", dest="inp", default=None, type=str, help="Input path.")
    parser.add_argument("-o", "--outpath", dest="outp", default=None, type=str, help="Output path.")
    return parser.parse_args()

if __name__ == "__main__":
    # read args
    args = get_args()

    # handle with args
    if args.inp is None:
        raise Exception("No input path.")
    if args.outp is None:
        raise Exception("No output path.") 
    if os.path.exists(args.outp) == False:
        print(f"exec mkdir -p {args.outp}")
        os.system(f"mkdir -p {args.outp}")

    # handle imgs
    imgs = os.listdir(args.inp)
    for img in imgs:
        print(f"Handling with {img}.")
        inpath = os.path.join(args.inp, img)
        outpath = os.path.join(args.outp, img)
        os.system(f"python simplify.py --img {inpath} --out {outpath} --model model_pencil2")