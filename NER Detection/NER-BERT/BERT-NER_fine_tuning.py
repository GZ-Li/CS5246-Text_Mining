from NERDA.models import NERDA

from NERDA.datasets import get_conll_data, download_conll_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# download_conll_data("./Pretrained_dataset")

model = NERDA(dataset_training = get_conll_data(split = 'train', dir = "Pretrained_dataset"),
              dataset_validation = get_conll_data('valid', dir = "Pretrained_dataset"),
              transformer = 'bert-base-multilingual-uncased',
              tag_scheme = [
                            'B-PER',
                            'I-PER', 
                            'B-ORG', 
                            'I-ORG', 
                            'B-LOC', 
                            'I-LOC', 
                            'B-MISC', 
                            'I-MISC',
                            'B-Med',
                            'I-Med',
                            'B-DATE',
                            'I-DATE',
                            'B-Fin',
                            'I-Fin'
                            ],
              hyperparameters = {'epochs': 1, 'warmup_steps': 500, 'train_batch_size': 64, 'learning_rate': 0.0001}
              )

# model = NERDA(dataset_training = get_conll_data(split = 'train'),
#               dataset_validation = get_conll_data('valid'),
#               transformer = 'cardiffnlp/twitter-xlm-roberta-base',
#               tag_scheme = [
#                             'B-PER',
#                             'I-PER', 
#                             'B-ORG', 
#                             'I-ORG', 
#                             'B-LOC', 
#                             'I-LOC', 
#                             'B-MISC', 
#                             'I-MISC',
#                             ],
#               hyperparameters = {'epochs': 16, 'warmup_steps': 500, 'train_batch_size': 32, 'learning_rate': 0.0001}
#               )

# text = "Mary has a good fitness!"
# print(model.predict_text(text))

# from NERDA.precooked import EN_BERT_ML
# model = EN_BERT_ML()
# model.download_network()
# model.load_network()

# text = "I need some medicine, including Paracetamol."
# print(model.predict_text(text))

model.train()

print(model.evaluate_performance(dataset = get_conll_data('test', dir = "Pretrained_dataset")))

# for name, param in model.named_parameters():
#     print(name, param.data)


# from NERDA.models import NERDA

# from NERDA.datasets import get_conll_data
# model = NERDA(dataset_training = get_conll_data('train'),
#               dataset_validation = get_conll_data('valid'),
#               transformer = 'bert-base-multilingual-uncased')

# model.train()

# text = "Chris is the best teacher I've ever met."
# print(model.predict_text(text))

# text = "Guanzhen is an outstanding student."
# print(model.predict_text(text))

# text = "Mary suffers from an arthritis."
# print(model.predict_text(text))

# text = "Guanzhen is a well-performed student."
# print(model.predict_text(text))

# text = "Li Guanzhen is a well performed student."
# print(model.predict_text(text))

# text = "Mary suffers from an arthritis."
# print(model.predict_text(text))

# text = "Mary suffers from an arthritis."
# print(model.predict_text(text))

# for name, param in model.network.named_parameters():
#     print(name, param.data)
    
# for name, _ in model.network.named_parameters():
#     print(name)

f = open('new_gen_fin.txt', encoding = 'gbk')

txt = []

for line in f:
    txt.append(line)

new_lst = []

for i in txt:
    if i != '\n':
        new_lst.append(i.split()[0])

corpus = ' '.join(new_lst)

sens = corpus.split(".")

# for sen in sens:
#     if sen != sens[-1]:
        
print(model.predict_text(sens[0]))