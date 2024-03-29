#!/usr/bin/env python

from __future__ import division, unicode_literals

import os
import argparse
import math
import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts
import txt_utils
import model_utils

import torchtext

def _report_rouge(result_file,golden_file):
    "1"
    #import sys
    #sys.path.append('../Lib/site-packages/rouge')
    #from rouge_wrap import RougeWrapper
    #r=RougeWrapper()
    #results = r.evaluate_for_pair_files(golden_file, result_file)
    "2"
    from rouge import Rouge
    #with open("pred.txt", "r") as f1, open("data/task1_ref0.txt", "r") as f2:
    #    rouge = Rouge()
    #    for l1, l2 in zip(f1, f2):
    #        scores = rouge.get_scores(l1, l2, avg=True)
    #print(scores)
    "3"
    from rouge import FilesRouge
    r = FilesRouge(result_file,golden_file)
    results = r.get_scores(avg=True)
    for k,v in results.items():
        if 'f' not in v:
            continue
        print(k,v)

def select_templates(src_file,tmp_file,score_file,output_file,tgt_file):
    current_src=''
    opt_template_scores=None
    with open(src_file,encoding='utf-8') as f_src,\
            open(tmp_file,encoding='utf-8') as f_tmp, \
            open(score_file,encoding='utf-8') as f_score,\
            open(output_file,'w',encoding='utf-8') as f_out:
        for line_src,line_tmp,line_score in zip(f_src,f_tmp,f_score):
            line_src=line_src.strip()
            if len(line_src)==0:
                continue
            line_tmp=line_tmp.strip()
            score=float()
            if line_src!=current_src:
                if opt_template_scores:
                    print(opt_template_scores[0],file=f_out)
                current_src=line_src
                opt_template_scores=[line_tmp,score]
            elif score>opt_template_scores[1]:
                opt_template_scores=[line_tmp,score]
        if opt_template_scores:
            print(opt_template_scores[0],file=f_out)
    _report_rouge(output_file,tgt_file)
    return

def main():
    parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)
    group = parser.add_argument_group('Rerank')
    group.add_argument('-templates', required=True,
                    help="Path to the test templates")
    opt = parser.parse_args()
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        model_utils.load_test_model(opt, dummy_opt.__dict__)

    fields["spliter_pos"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False) 

    # Unfold templates
    src_path,tmp_path=txt_utils.unfold_templates(opt.src, opt.templates)
    
    # Test data
    data=txt_utils.build_template_dataset(fields, src_path, None, tmp_path,use_filter_pred=False,with_pos=True,dynamic_dict=False)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    count=0
    #offset=0
    #scores=[]
    score_dict={}
    for batch in data_iter:
        #print(batch.indices)
        #index=batch.indices-offset
        src = onmt.io.make_features(batch, 'src', 'text')
        predict_score=model.predict_rouge(src,batch.src[1],batch.spliter_pos)
        #ordered_score=predict_score[index].data
        #scores.extend(ordered_score)
        #offset+=index.size(0)
        for index,score in zip(batch.indices.data,predict_score.data):
            score_dict[int(index)]=float(score)
        count+=1
        if count% 100==0:
            print('score {} batches'.format(count))
        #if count>10: break
        
    # File to write sentences to.
    score_file = opt.output + '.score'
    print('score_file is ' + score_file)
    print('opt.tgt is ' + opt.tgt)
    out_file = open(score_file, 'w', encoding='utf-8')
    print(len(score_dict))
    for index in range(len(score_dict)):
        print(score_dict[index], file=out_file)
    out_file.close()
    select_templates(src_path,tmp_path,score_file,opt.output,opt.tgt)
    

if __name__ == "__main__":
    main()
    print('finished')
