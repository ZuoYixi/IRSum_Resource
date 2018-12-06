The update of opennmt overwrited a part of our code and made the model unable to run. We are trying to fix these bugs.

# train step：
step1:pair_preprocess.py
step2:pair_train.py
step3:pair_rerank.py
step4:pair_translate.py

1.python pair_preprocess.py (arguments)
-train_src data/train.article.txt
-train_tgt data/train.title.txt
-valid_src data/valid.article.filter.txt
-valid_tgt data/valid.title.filter.txt
-save_data outputData/preprocess.output
-train_template data/train.title.txt
-valid_template data/valid.title.filter.txt

2.python pair_train.py (argument)
-data outputData/preprocess.output (-gpuid 0)

3.python pair_rerank.py (arguments)
-model model/model_acc_0.00_ppl_57.02_e13.pt
-src data/input.txt
-templates data/test.rerank.template
-tgt data/task1_ref0.txt