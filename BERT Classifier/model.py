from transformers import Trainer, TrainingArguments,BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class BERT_CLASSIFIER(nn.Module):
    def __init__(self, num_outputs, hidden):
        super(BERT_CLASSIFIER, self).__init__()
        self.num_labels = num_outputs
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        self.dropout = nn.Dropout(0.35)
        self.mlp = nn.Linear(768, hidden)
        self.classifier = nn.Linear(hidden, num_outputs)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=True,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(self.mlp(pooled_output))

        return logits
    def get_attention(self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        attentions = outputs.attentions
        return_attention = None
        for attention in attentions[-1]:
            normalized_attention = torch.softmax(attention, dim=2).squeeze().detach().numpy()
            return_attention = normalized_attention

        return return_attention

class MLP(nn.Module):
    def __init__(self, num_outputs, hidden_size):
        super(MLP, self).__init__() 
        self.num_labels = num_outputs
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.out = nn.Linear(300, num_outputs, bias=False)
    def forward(self,x):
        embedding = self.embedding(x)
        embedding = torch.mean(embedding, axis = 1)
        out = self.out(embedding)
        return out 

class LSTM(nn.Module):
    def __init__(self, num_outputs, hidden_size, num_layers):
        super(LSTM, self).__init__() 
        self.num_labels = num_outputs
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.lstm = nn.LSTM(300, hidden_size, num_layers=num_layers, batch_first=True , bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_size*2, hidden_size, bias=False)

    def forward(self,x):
        embedding = self.embedding(x)
        out, (h_state, c_state) = self.lstm(embedding)
        hidden_feature = out[:, -1, :]
        return self.out(self.dropout(hidden_feature))


class CNN(nn.Module):
    def __init__(self, num_outputs, hidden_size):
        super(CNN, self).__init__() 
        self.num_labels = num_outputs
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5
        self.seq_len = 27
        self.stride = 1
        self.out_size = 1
        self.num_labels = num_outputs
        
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)
        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)
      
        # Fully connected layer definition
        self.fc = nn.Linear(1180, self.num_labels)
    def forward(self, x):
        x = self.embedding(x)
      
        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)
        # print(union.shape)
        out = self.fc(union)
        
        return out 
