from model import BERT_CLASSIFIER
from parse import parse_args
from data import get_data, Recorder
from transformers import AdamW
from utils import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.cuda)
    train_data = get_data(args, args.max_len, args.batch_size, type = 'train')
    val_data = get_data(args, args.max_len, args.batch_size, type = 'val')
    test_data = get_data(args, args.max_len, args.batch_size, type='test')
    recorder = Recorder()
    saveID = args.saveID
    checkpoint_buffer = []
    base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)
    run_path = './runs/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    ensureDir(base_path)
    ensureDir(run_path)

    if args.modeltype == 'Bert_Classifier':
        model = BERT_CLASSIFIER(2, args.hidden)
    else:
        model = None

    model.cuda(device)
    flag = False
    optim = AdamW(model.parameters(), lr=2 * 1e-5)

    loss_fct = torch.nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        if flag:
            break
        running_loss = 0
        num_batch = 0
        pbar = tqdm(enumerate(train_data), total=len(train_data))
        for batch_i, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fct(outputs, labels)

            # clear all weight gradients before backtrack
            optim.zero_grad()
            # backtrack
            loss.backward()
            optim.step()
            running_loss += loss.detach().item()
            num_batch += 1

            model.eval()
            is_best, temp_flag = evaluation(args,val_data, model, epoch, recorder, device, base_path)
            if is_best:
                checkpoint_buffer = save_checkpoint(model, epoch, base_path, checkpoint_buffer, args.max2keep)
            if temp_flag:
                flag = True
            model.train()

        perf_str = ' Epoch %d train [%.5fs]' % (
            epoch, running_loss/num_batch)
        with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(perf_str + "\n")

        

    # Get result
    model = restore_best_checkpoint(recorder.best_valid_epoch, model, base_path, device)
    print_str = "The best epoch is % d" % recorder.best_valid_epoch
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")

    evaluation(args, test_data, model, epoch, recorder, device, base_path, name="test")



