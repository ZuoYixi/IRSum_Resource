import io
import sys
from onmt.io import TextDataset
from onmt.io.DatasetBase import ONMTDatasetBase, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.Utils import aeq
from itertools import chain
import torch
import torchtext
import codecs
from collections import Counter
from nltk.util import everygrams
import warnings

TEMPLATE_SPLITER='<t>'

class ShardedPairIterator(object):
    """
    No features
    """
    def __init__(self, src_path,tgt_path,template_path, line_truncate, shard_size,max_gram=2):
        """
        Args:
            corpus_path: the corpus file path.
            line_truncate: the maximum length of a line to read.
                            0 for unlimited.
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
#         try:
#             # The codecs module seems to have bugs with seek()/tell(),
#             # so we use io.open().
#             self.corpus = io.open(src_path, "r", encoding="utf-8")
#             self.tgt=io.open(tgt_path, "r", encoding="utf-8")
#             self.template = io.open(template_path, "r", encoding="utf-8")
#         except IOError:
#             sys.stderr.write("Failed to open corpus file: %s" % src_path)
#             sys.exit(1)
        self.src_path=src_path
        self.tgt_path=tgt_path
        self.template_path=template_path
        self.line_truncate = line_truncate
        self.shard_size = shard_size
        #self.last_pos = 0
        self.line_index = -1
        self.eof = False
        self.max_gram=max_gram
        self.n_feats=0

    def get_batches(self):
        batch=[]
        with open(self.src_path,encoding='utf-8') as f_src,open(self.tgt_path,encoding='utf-8') as f_tgt,open(self.template_path,encoding='utf-8') as f_template:
            for line_s,line_t,template in zip(f_src,f_tgt,f_template):
                line_s=line_s.split()
                line_t=line_t.split()
                template=template.split()
                if len(line_s)==0 or len(line_t)==0: continue
                self.line_index+=1
                example_dict=self._example_dict_iter(line_s, line_t, template)
                batch.append(example_dict)              
                if len(batch) >= self.shard_size != 0:
                    yield batch
                    batch=[]
        if len(batch)>0:
            yield batch
        return
# 
#     def __iter__(self):
#         """
#         Iterator of (example_dict, nfeats).
#         On each call, it iterates over as many (example_dict, nfeats) tuples
#         until this shard's size equals to or approximates `self.shard_size`.
#         """
#             # Yield tuples util this shard's size reaches the threshold.
#         self.corpus.seek(self.last_pos)
#         self.template.seek(self.last_pos)
#         self.tgt.seek(self.last_pos)
#         while True:
#             if self.shard_size != 0 and self.line_index % 64 == 0:
#                     # This part of check is time consuming on Py2 (but
#                     # it is quite fast on Py3, weird!). So we don't bother
#                     # to check for very line. Instead we chekc every 64
#                     # lines. Thus we are not dividing exactly per
#                     # `shard_size`, but it is not too much difference.
#                 cur_pos = self.corpus.tell()
#                 if cur_pos >= self.last_pos + self.shard_size:
#                     self.last_pos = cur_pos
#                     return
# 
#             line = self.corpus.readline()
#             template=self.template.readline()
#             line_t=self.tgt.readline()
#             if line == '':
#                 self.eof = True
#                 self.corpus.close()
#                 self.template.close()
#                 self.tgt.close()
#                 return
#             self.line_index += 1
#             line=line.strip()
#             line_t=line_t.strip()
#             if len(line)==0 or len(line_t)==0:
#                 continue
#             yield self._example_dict_iter(line,line_t,template)
# 
#     def hit_end(self):
#        return self.eof

    @property
    def num_feats(self):
        return 0

    def _example_dict_iter(self, line_s,line_t,template):
        score=self._compute_rouge(line_t, template)
        if self.line_truncate:
            line_s = line_s[:self.line_truncate]
        spliter_pos=len(line_s)    
        words_in=line_s+[TEMPLATE_SPLITER]+template
        example_dict={}
        example_dict['src']=words_in
        example_dict['indices']=self.line_index
        example_dict['spliter_pos']=spliter_pos
        example_dict['tgt']=line_t
        example_dict['rouge_score']=score
        return example_dict
    
    def _compute_rouge(self,words_t,words_c):
        ngrams_t=set(everygrams(words_t, max_len=self.max_gram))
        ngrams_c=everygrams(words_c,max_len=self.max_gram)
        match_count=0
        total_count=0
        for ngram_c in ngrams_c:
            total_count+=1
            if ngram_c in ngrams_t:
                match_count+=1
        if total_count==0:
            warnings.warn('empty template for title:{}'.format(' '.join(words_t)))
            return 0
        score=match_count/total_count
        return score

class ShardedTemplateTextCorpusIterator(object):
    """
    This is the iterator for text corpus, used for sharding large text
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the corpus file into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """
    def __init__(self, corpus_path,template_path, line_truncate, side, shard_size,
                 assoc_iter=None):
        """
        Args:
            corpus_path: the corpus file path.
            line_truncate: the maximum length of a line to read.
                            0 for unlimited.
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
            self.template = io.open(template_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)

        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                if line == '':
                    raise AssertionError(
                        "Two corpuses must have same number of lines!")
                template=self.template.readline()
                self.line_index += 1
                line=' '.join([line,TEMPLATE_SPLITER,template])
                yield self._example_dict_iter(line)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
                self.template.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            self.template.seek(self.last_pos)
            while True:
                if self.shard_size != 0 and self.line_index % 64 == 0:
                    # This part of check is time consuming on Py2 (but
                    # it is quite fast on Py3, weird!). So we don't bother
                    # to check for very line. Instead we chekc every 64
                    # lines. Thus we are not dividing exactly per
                    # `shard_size`, but it is not too much difference.
                    cur_pos = self.corpus.tell()
                    if cur_pos >= self.last_pos + self.shard_size:
                        self.last_pos = cur_pos
                        raise StopIteration

                line = self.corpus.readline()
                template=self.template.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    self.template.close()
                    raise StopIteration

                self.line_index += 1
                line=' '.join([line,TEMPLATE_SPLITER,template])
                yield self._example_dict_iter(line)

    def hit_end(self):
        return self.eof

    @property
    def num_feats(self):
        # We peek the first line and seek back to
        # the beginning of the file.
        saved_pos = self.corpus.tell()

        line = self.corpus.readline().split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        _, _, self.n_feats = TextDataset.extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats

    def _example_dict_iter(self, line):
        line = line.split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        words, feats, n_feats = TextDataset.extract_text_features(line)
        example_dict = {self.side: words, "indices": self.line_index}
        if feats:
            # All examples must have same number of features.
            aeq(self.n_feats, n_feats)

            prefix = self.side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))

        return example_dict
    
    
class AdvancedTextDataset(ONMTDatasetBase):
    def __init__(self, fields, examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 dynamic_dict=True):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)
        examples_iter=iter(examples_iter)
        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = [self._construct_example_fromlist(
                            ex_values, out_fields)
                        for ex_values in example_values]
        super(AdvancedTextDataset, self).__init__(
            out_examples, out_fields
        )
        #print('out examples:',len(self.examples))
    
    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    scores[:, b, ti] += scores[:, b, offset + i]
                    scores[:, b, offset + i].fill_(1e-20)
        return scores

    @staticmethod
    def make_template_examples_nfeats_tpl(src_path,template_path, truncate, side,with_pos=False):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'tgt']

        if src_path is None:
            return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            AdvancedTextDataset.read_template_file(src_path,template_path, truncate, side,with_pos )

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)




    @staticmethod
    def build_example_from_pair(truncate, side, i, line, line_t,with_pos=False):
        words_s=line.split()
        if truncate:
            #line = line[:truncate]
            words_s=words_s[:truncate]
        spliter_pos=len(words_s)
        line=' '.join(words_s)
        line = ' '.join([line, TEMPLATE_SPLITER, line_t]).split()

        words, feats, n_feats = TextDataset.extract_text_features(line)
        example_dict = {side:words, "indices":i}
        if with_pos:
            example_dict['spliter_pos']=spliter_pos
        if feats:
            prefix = side + "_feat_"
            example_dict.update((prefix + str(j), f) for (j, f) in enumerate(feats))
        return example_dict, n_feats

    @staticmethod
    def read_template_file(src_path,template_path, truncate, side,with_pos=False):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        with open(src_path, encoding= "utf-8") as corpus_file,open(template_path,encoding='utf-8') as template_file:
            for i, (line,line_t) in enumerate(zip(corpus_file,template_file)):
                line = line.strip()
                if len(line)==0: continue
                line_t=line_t.strip()
                example_dict, n_feats = AdvancedTextDataset.build_example_from_pair(truncate, side, i, line, line_t,with_pos)
                yield example_dict, n_feats

    @staticmethod
    def get_fields():
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)


        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        def make_src(data, vocab, is_train):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab, is_train):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)
        
        fields["spliter_pos"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)        

        fields["rouge_score"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            sequential=False)  
        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            _, _, num_feats = TextDataset.extract_text_features(f_line)

        return num_feats

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src))
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                        [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example
            
def build_template_dataset(fields, src_path, tgt_path,template_path,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, use_filter_pred=True,with_pos=False):

    # Build src/tgt examples iterator from corpus files, also extract
    # number of features.
    src_examples_iter, num_src_feats = \
            AdvancedTextDataset.make_template_examples_nfeats_tpl(
                src_path,template_path, src_seq_length_trunc, "src",with_pos)

    # For all data types, the tgt side corpus is in form of text.
    tgt_examples_iter, num_tgt_feats = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_path, tgt_seq_length_trunc, "tgt")

    dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter,
                          num_src_feats, num_tgt_feats,
                          src_seq_length=src_seq_length,
                          tgt_seq_length=tgt_seq_length,
                          dynamic_dict=dynamic_dict,
                          use_filter_pred=use_filter_pred)

    return dataset


def unfold_templates(src_path, 
                              templates_path
                  ):

    # Build src/tgt examples iterator from corpus files, also extract
    # number of features.
    repeat_src_path=src_path+'.repeat'
    #repeat_tgt_path=tgt_path+'.repeat'
    unfold_template_path=templates_path+'.repeat'
    with open(src_path,encoding='utf-8') as f_s, open(templates_path,encoding='utf-8') as f_tmp, \
        open(repeat_src_path,'w',encoding='utf-8') as f_rs, \
        open(unfold_template_path,'w',encoding='utf-8') as f_rtmp:
        for line_s,line_tmp in zip(f_s,f_tmp):
            line_s=line_s.strip()
            if len(line_s)==0: continue
            line_tmp=line_tmp.strip()
            templates=line_tmp.split(' ||| ')
            for template in templates:
                print(line_s,file=f_rs)
                print(template,file=f_rtmp)
    return repeat_src_path,unfold_template_path
    

from collections import defaultdict
def load_fields_from_vocab(vocab):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    fields = AdvancedTextDataset.get_fields()
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields
