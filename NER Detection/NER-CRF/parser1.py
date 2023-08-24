import argparse

def parse_args():

    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--modeltype', type=str, default='BertCrf',
                        help='Specify model save path.')
    parser1.add_argument('--cuda', type=bool, default=True,
                        help='Whether use gpu or not.')
    parser1.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser1.add_argument('--num_epochs', type=int, default=2,
                        help='num_epochs')
    parser1.add_argument('--num_labels', type = int, default = 20, help = 'Num of ner labels.')
    parser1.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate for learnable parameters.')
    parser1.add_argument('--weight_decay', type = float, default = 0.0001, help = 'Weight decay.')

    return parser1.parse_args()
