import os
import sys
import argparse
from model import Model

def set_parameters(argv):
    parser=argparse.ArgumentParser(description="Manual settings for this model")

    # General settings
    parser.add_argument('--gpu', type=str, default=None)  # None for CPU
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test', 'demo'])
    parser.add_argument('--dataset', type=str, default="multipie", choices=['multipie', 'kitti',
                                    'multipie_larger', 'multipie_all',
                                    'multipie_asym_lr', 'multipie_asym_l', 'multipie_asym_r', 'multipie_asym'])

    # Hyper-parameters
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--gen_lr', type=float, default=0.0001)
    parser.add_argument('--disc_lr', type=float, default=0.00001)
    parser.add_argument('--w_vc', type=float, default=0.01)
    parser.add_argument('--w_adv', type=float, default=0.001)
    parser.add_argument('--w_sharp', type=float, default=0.01)
    parser.add_argument('--sharp_loss', type=str, default='l1')

    # Training settings
    parser.add_argument("--save_folder", type=str, default='run')  # folder for saving checkpoints

    # Testing settings
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default='results')  # folder for saving testing results

    # Demo settings
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--input_l_name", type=str, default=None)
    parser.add_argument("--input_r_name", type=str, default=None)
    parser.add_argument("--output_name", type=str, default='result')


    return parser.parse_args()

def main(argv):
    args = set_parameters(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    Model(args).run()

if __name__ == "__main__":
    main(sys.argv)