from parser1 import parse_args
from transformers import BertModel, BertPreTrainedModel, BertConfig, BertTokenizer
from data import data
from model import BertCrfForNer
import matplotlib.pyplot as plt
import transformers
import torch
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    args = parse_args()
    plm_path_dict = {"BertCrf": 'data/bert_base_uncased'}
    if args.modeltype in list(plm_path_dict.keys()):
        pretrain_model_path = plm_path_dict[args.modeltype]
    else:
        raise RuntimeError("No such models defined!!!")

    num_epoch = args.num_epochs
    batch_size = args.batch_size
    num_labels = args.num_labels # Must be consistent with the input data's size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    min_loss = 10000000
    train_loss = []
    valid_loss = []
    train_acc = []
    train_recall = []

    train_loader, test_loader, valid_loader = data(batch_size)

    config = BertConfig.from_pretrained(pretrain_model_path, num_labels=num_labels)  # num_labels represents the number of NEs.
    model = BertCrfForNer(config)

    if torch.cuda.is_available() and args.cuda:
        print("Running on GPU!")
        model.to('cuda')
        #train_loader.to('cuda')
        #test_loader.to('cuda')
        #valid_loader.to('cuda')

    for epoch in range(num_epoch):

        model.train()
        for step, (train_input_ids, train_token_type_ids, train_attention_mask, train_labels) in enumerate(train_loader):
            if torch.cuda.is_available() and args.cuda:
                train_input_ids = train_input_ids.to("cuda")
                train_token_type_ids = train_token_type_ids.to('cuda')
                train_attention_mask = train_attention_mask.to('cuda')
                train_labels = train_labels.to('cuda')
            loss, outputs = model(input_ids=train_input_ids, attention_mask=train_attention_mask, token_type_ids=train_token_type_ids, labels=train_labels)

            # Define parameters need training
            no_bert = ["word_embedding_adapter", "word_embeddings", "classifier", "crf"]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                # bert no_decay
                {
                    "params": [p for n, p in model.named_parameters()
                               if (not any(
                            nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and any(
                            nd in n for nd in no_decay)],
                    "weight_decay": 0.0, 'lr': 0.0
                },
                # bert decay
                {
                    "params": [p for n, p in model.named_parameters()
                               if (not any(
                            nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and not any(
                            nd in n for nd in no_decay)],
                    "weight_decay": weight_decay, 'lr': 0.0
                },
                # other no_decay
                {
                    "params": [p for n, p in model.named_parameters()
                               if
                               any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and any(
                                   nd in n for nd in no_decay)],
                    "weight_decay": 0.0, "lr": learning_rate
                },
                # other decay
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in
                                   no_bert) and n != 'bert.embeddings.word_embeddings.weight' and not any(
                                   nd in n for nd in no_decay)],
                    "weight_decay": weight_decay, "lr": learning_rate
                }
            ]

            outputs_labels = torch.max(outputs, axis = 2)[1]
            num_true_pred = (outputs_labels == train_labels).sum().item()
            accuracy = num_true_pred / train_labels.nelement()

            ner_labels = train_labels != 0
            true_ners = (outputs_labels == train_labels) & ner_labels
            num_ners = ner_labels.sum().item()
            recall = true_ners.sum().item() / num_ners

            optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=0.01, eps=0.000001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch: {}, Step: {}, Loss: {}, Accuracy:{}, Recall: {}".format(epoch, step, loss, accuracy, recall))

            train_loss.append(loss.item())
            train_acc.append(accuracy)
            train_recall.append(recall)

            model.eval()
            for eval_step, (valid_input_ids, valid_token_type_ids, valid_attention_mask, valid_labels) in enumerate(valid_loader):

                if torch.cuda.is_available() and args.cuda:
                    valid_input_ids = valid_input_ids.to('cuda')
                    valid_token_type_ids = valid_token_type_ids.to('cuda')
                    valid_attention_mask = valid_attention_mask.to('cuda')
                    valid_labels = valid_labels.to('cuda')

                eval_loss = 0
                temp_eval_loss, eval_outputs = model(input_ids=valid_input_ids, attention_mask=valid_attention_mask,
                                      token_type_ids=valid_token_type_ids, labels=valid_labels)
                eval_loss += temp_eval_loss
            if eval_loss < min_loss:
                print("original loss: {}, new loss: {}".format(min_loss, eval_loss))
                min_loss = eval_loss
                torch.save(model.state_dict(), "model.pth")
                print("save")
            valid_loss.append(eval_loss.item())

    plt.plot(train_loss)
    plt.xlabel("epochs")
    plt.ylabel("train_loss")
    plt.title("train loss v.s. epochs")
    plt.savefig("train_loss.jpg")

    plt.plot(train_acc)
    plt.xlabel("epochs")
    plt.ylabel("train_accuracy")
    plt.title("train accuracy v.s. epochs")
    plt.savefig("train_accuracy.jpg")

    plt.plot(train_recall)
    plt.xlabel("epochs")
    plt.ylabel("train_recall")
    plt.title("train recall v.s. epochs")
    plt.savefig("train_recall.jpg")

    plt.plot(valid_loss)
    plt.xlabel("epochs")
    plt.ylabel("valid loss")
    plt.title("valid loss v.s. epochs")
    plt.savefig("valid_loss.jpg")

    # Calculate Metrics
    model.eval()
    test_loss = 0
    test_acc = 0
    test_recall = 0
    num_true_pred = 0
    num_true_ner = 0
    test_num = 0
    ner_num = 0
    for test_step, (test_input_ids, test_token_type_ids, test_attention_mask, test_labels) in enumerate(test_loader):

        if torch.cuda.is_available() and args.cuda:
            test_input_ids = test_input_ids.to('cuda')
            test_token_type_ids = test_token_type_ids.to('cuda')
            test_attention_mask = test_attention_mask.to('cuda')
            test_labels = test_labels.to('cuda')

        temp_test_loss, test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids, labels=test_labels)
        test_loss += temp_test_loss

        test_outputs_labels = torch.max(test_outputs, axis=2)[1]
        temp_num_true_pred = (test_outputs_labels == test_labels).sum().item()
        test_num += test_labels.nelement()
        num_true_pred += temp_num_true_pred

        ner_labels = test_labels != 0
        true_ners = (test_outputs_labels == test_labels) & ner_labels
        num_true_ner += true_ners.sum().item()
        ner_num += ner_labels.sum().item()

    test_acc = num_true_pred / test_num
    test_recall = num_true_ner / ner_num

    print("Test Results: \nLoss: {}\nAccuracy: {}\nRecall: {}".format(test_loss, test_acc, test_recall))
