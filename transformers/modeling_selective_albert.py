import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from .modeling_albert import AlbertPreTrainedModel, AlbertModel


class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def init_HP(self, args):
        self.lamb = args.lamb

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        routine='raw',
        **kwargs
    ):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        batch_size = logits.shape[0]
        device = logits.device

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

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

        return outputs  # (loss), logits, (hidden_states), (attentions)
