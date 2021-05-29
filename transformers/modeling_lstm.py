from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F


from .modeling_bert import BertPreTrainedModel, BertLayerNorm, BERT_PRETRAINED_MODEL_ARCHIVE_MAP

BERT_PRETRAINED_MODEL_ARCHIVE_MAP['lstm-base'] = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin"


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.dropout(self.LayerNorm(inputs_embeds))
        return embeddings


class LstmForSequenceClassification(BertPreTrainedModel):
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super(LstmForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_layers
        self.max_pooling = config.max_pooling

        self.embeddings = BertEmbeddings(config)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_prob,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(2*config.hidden_size, config.num_labels)

        self.init_weights()

    def init_HP(self, args):
        self.lamb = args.lamb

    def apply_lstm(self, input_ids):
        # input id input_ids
        # output is the complete output of lstm
        # for efficiency: improve this function (sorting? packing?)
        batch_size, max_seq_len = input_ids.shape
        sent_len = torch.sum(input_ids!=0, dim=1)
        sorted_sent_len, forward_sort_order = torch.sort(sent_len, descending=True)
        _, backward_sort_order = torch.sort(forward_sort_order)
        sorted_batch = self.embeddings(input_ids)[forward_sort_order]
        packed_batch = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_batch, sorted_sent_len.cpu(), batch_first=True
        )
        complete_output, (h_n, _) = self.lstm(packed_batch)
        if self.max_pooling:
            output = torch.nn.utils.rnn.pad_packed_sequence(complete_output)[0]  # basically unpack it
            output = output.max(dim=0)[0][backward_sort_order]
            # max: max pooling along the seq_len dimension
            # [backward_order]: recover the order before sorting
        else:
            output = h_n.permute(1,0,2)[backward_sort_order].view(batch_size, self.num_layers, 2, -1)[:,-1,:,:].view(batch_size, -1)
            # permute: change batch to the first index
            # [backward_order]: recover the order before sorting
            # view: separate num_layers with num_directions
            # [...]: take only the last layer
            # view: merge both directions
        return output  # shape: [batch_size, 2*hidden_size]

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                routine='raw',
                **kwargs):

        lstm_output = self.apply_lstm(input_ids)
        logits = self.classifier(self.dropout(lstm_output))

        outputs = (logits,)
        batch_size = logits.shape[0]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if routine == 'raw':
                pass
            elif routine == 'reg-curr':
                if self.training:
                    # here correctness is always 0 or 1
                    confidence, prediction = torch.softmax(logits, dim=1).max(dim=1)
                    correctness = (prediction == labels)
                    correct_confidence = torch.masked_select(confidence, correctness)
                    wrong_confidence = torch.masked_select(confidence, ~correctness)
                    regularizer = 0
                    for cc in correct_confidence:
                        for wc in wrong_confidence:
                            regularizer += torch.clamp(wc-cc, min=0) ** 2
                    loss += self.lamb * regularizer
            elif routine == 'reg-hist':
                if self.training:
                    # here correctness is continuous in [0,1]
                    confidence, _prediction = torch.softmax(logits, dim=1).max(dim=1)
                    correctness = kwargs['history_record']
                    _, sorted_correctness_index = torch.sort(correctness)
                    lower_index = sorted_correctness_index[:int(0.2 * batch_size)]
                    higher_index = sorted_correctness_index[int(0.2 * batch_size):]
                    regularizer = 0
                    for li in lower_index:  # indices with lower correctness
                        for hi in higher_index:
                            if correctness[li] < correctness[hi]:
                                # only if it's strictly smaller
                                regularizer += torch.clamp(
                                    confidence[li] - confidence[hi], min=0
                                ) ** 2
                    loss += self.lamb * regularizer
            else:
                raise NotImplementedError()
            outputs = (loss,) + outputs

        return outputs  # loss, logits
