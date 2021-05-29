from __future__ import absolute_import, division, print_function, unicode_literals

import json
import sys
from io import open

from .configuration_utils import PretrainedConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP['lstm-base'] = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"


class LstmConfig(PretrainedConfig):
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 lstm_hidden=200,
                 num_layers=2,
                 dropout_prob=0.1,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 max_pooling=True,
                 **kwargs):
        super(LstmConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout_prob = dropout_prob
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.lstm_hidden = lstm_hidden
            self.max_pooling = max_pooling
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")
