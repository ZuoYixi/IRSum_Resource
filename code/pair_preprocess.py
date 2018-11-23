#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys

import torch
import txt_utils
import onmt.io
import onmt.opts as opts


def check_existing_pt_files(opt):
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup exisiting pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='template_preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)
    group = parser.add_argument_group('Template')
    group.add_argument('-train_template', required=True,
                       help="Path to the training template")
    group.add_argument('-valid_template', required=True,
                       help="Path to the valid template")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def build_save_text_dataset_in_shards(src_corpus, tgt_corpus,template_corpus, fields,
                                      corpus_type, opt):
    '''
    Divide the big corpus into shards, and build dataset separately.
    This is currently only for data_type=='text'.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.

    To tackle this, we only read in part of the corpus file of size
    `max_shard_size`(actually it is multiples of 64 bytes that equals
    or is slightly larger than this size), and process it into dataset,
    then write it to disk along the way. By doing this, we only focus on
    part of the corpus at any moment, thus effectively reducing memory use.
    According to test, this method can reduce memory footprint by ~50%.

    Note! As we process along the shards, previous shards might still
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.

    If `max_shard_size` is 0 or is larger than the corpus size, it is
    effectively preprocessed into one dataset, i.e. no sharding.
    '''

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % src_corpus)

    ret_list = []
    pair_iter=txt_utils.ShardedPairIterator(src_corpus,tgt_corpus,template_corpus,opt.src_seq_length_trunc,opt.max_shard_size)

    print(' * divide corpus into shards and build dataset separately'
          '(shard_size = %d lines).' % opt.max_shard_size)

    index = 0
    for batch in pair_iter.get_batches():
        index += 1
        dataset=txt_utils.AdvancedTextDataset(fields,batch,
                                              dynamic_dict=opt.dynamic_dict)
        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
                opt.save_data, corpus_type, index)
        print(" * saving train data shard to %s." % pt_file)
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
        template_corpus=opt.train_template
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt
        template_corpus=opt.valid_template

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    if opt.data_type == 'text':
        return build_save_text_dataset_in_shards(
                src_corpus, tgt_corpus,template_corpus, fields,
                corpus_type, opt)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = onmt.io.build_dataset(
                fields, opt.data_type, src_corpus, tgt_corpus,
                src_dir=opt.src_dir,
                src_seq_length=opt.src_seq_length,
                tgt_seq_length=opt.tgt_seq_length,
                src_seq_length_trunc=opt.src_seq_length_trunc,
                tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
                dynamic_dict=opt.dynamic_dict,
                sample_rate=opt.sample_rate,
                window_size=opt.window_size,
                window_stride=opt.window_stride,
                window=opt.window)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    print(" * saving train dataset to %s." % pt_file)
    torch.save(dataset, pt_file)

    return [pt_file]


def build_save_vocab(train_dataset, fields, opt):
    fields = onmt.io.build_vocab(train_dataset, fields, opt.data_type,
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)


def main():
    opt = parse_args()

    print("Loading Fields object...")
    fields = txt_utils.AdvancedTextDataset.get_fields()

    print("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    print("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)
    print('Finish')


if __name__ == "__main__":
    main()
