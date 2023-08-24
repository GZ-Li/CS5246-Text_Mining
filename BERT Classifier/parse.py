import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeltype', type=str, default='Bert_Classifier',
                        help='Specify model save path.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--max_len', type=int, default=16,
                        help='max len for bert')
    parser.add_argument('--hidden', type=int, default=64,
                        help='hidden for MLP')
    parser.add_argument('--batch_size', type=int, default=15,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='num_epochs')
    parser.add_argument('--saveID', type=str, default="",
                        help='Specify model save path.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in data loader')
    parser.add_argument('--dataset', nargs='?', default='tweets',
                        help='Choose a dataset')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping point.')
    parser.add_argument('--max2keep', type=int, default=10,
                        help='max checkpoints to keep')
    
    return parser.parse_args()
