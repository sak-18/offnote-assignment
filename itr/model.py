import torch
import numpy as np
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from config import replace, preEnc, preEncDec
from easydict import EasyDict as ED
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
from data import IndicDataset, PadSequence


def init_seed():
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


class TranslationModel(LightningModule):

    def __init__(self, encoder, decoder, tokenizers, pad_sequence):

        super().__init__()
        # Creating encoder and decoder with their respective embeddings.
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizers = tokenizers
        self.pad_sequence = pad_sequence
        self.config = preEncDec

    def forward(self, encoder_input_ids, decoder_input_ids):

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states,
                                    masked_lm_labels=decoder_input_ids)

        return loss, logits

    def prepare_data(self):
        self.indic_train = IndicDataset(
            self.tokenizers.src, self.tokenizers.tgt, self.config.data, True)
        self.indic_test = IndicDataset(
            self.tokenizers.src, self.tokenizers.tgt, self.config.data, False)

    def train_dataloader(self):
        train_loader = DataLoader(self.indic_train,
                                  batch_size=self.config.batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=self.pad_sequence)
        self.len_train_loader = len(train_loader)
        return train_loader

    def val_dataloader(self):
        eval_loader = DataLoader(self.indic_test,
                                 batch_size=self.config.eval_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=self.pad_sequence)
        self.len_eval_loader = len(eval_loader)
        return eval_loader

    def configure_optimizers(self):
        init_seed()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        loss, logits = self(data, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        loss, logits = self(data, target)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
