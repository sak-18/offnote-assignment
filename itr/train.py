from config import replace, preEnc, preEncDec
import numpy as np
import torch
from data import IndicDataset, PadSequence
from time import time
from pathlib import Path
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import TranslationModel
from pytorch_lightning import Trainer
from easydict import EasyDict as ED
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

checkpoint_callback = ModelCheckpoint(
    filepath='./pretrained-model/best-model.ckpt',
    verbose=True,
    monitor='val_loss',
    mode='min'
)


def build_enc_dec_tokenizers(config):

    src_tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'

    # hidden_size and intermediate_size are both wrt all the attention heads.
    # Should be divisible by num_attention_heads
    encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)

    decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                is_decoder=False)  # CHANGE is_decoder=True

    # Create encoder and decoder embedding layers.
    encoder_embeddings = torch.nn.Embedding(
        src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
    decoder_embeddings = torch.nn.Embedding(
        tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

    encoder = BertModel(encoder_config)
    # encoder.set_input_embeddings(encoder_embeddings.cuda())
    encoder.set_input_embeddings(encoder_embeddings)  # 1

    decoder = BertForMaskedLM(decoder_config)
    # decoder.set_input_embeddings(decoder_embeddings.cuda())
    decoder.set_input_embeddings(decoder_embeddings)  # 2

    # model.cuda()

    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})
    return encoder, decoder, tokenizers


def preproc_data():
    from data import split_data
    split_data('../data/hin-eng/hin.txt', '../data/hin-eng')


def gen_model_loaders(config):
    encoder, decoder, tokenizers = build_enc_dec_tokenizers(config)
    pad_sequence = PadSequence(
        tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)

    return encoder, decoder, tokenizers, pad_sequence


def main(args):
    rconf = preEncDec
    encoder, decoder, tokenizers, pad_sequence = gen_model_loaders(
        rconf)

    model = TranslationModel(encoder=encoder, decoder=decoder,
                             tokenizers=tokenizers, pad_sequence=pad_sequence)

    writer = SummaryWriter(rconf.log_dir)

    if(args["modelpretrained"]):
        trainer = Trainer(checkpoint_callback=checkpoint_callback,
                          resume_from_checkpoint=args["modelpretrained"])
    else:
        trainer = Trainer()

    trainer.fit(model)
    model.save(tokenizers, rconf.model_output_dirs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelpretrained", required=False,
                    help="Path to the pretrained model")
    args = vars(ap.parse_args())
    preproc_data()
    main(args)
