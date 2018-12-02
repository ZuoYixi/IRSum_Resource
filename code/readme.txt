The update of opennmt overwrited a part of our code and made the model unable to run. We are trying to fix these bugs.

# 训练的流程：
step1:pair_preprocess.py
step2:pair_train.py
step3:pair_rerank.py
step4:pair_translate.py

python pair_preprocess.py 执行命令后面添加以下arguments
-train_src data/train.article.txt
-train_tgt data/train.title.txt
-valid_src data/valid.article.filter.txt
-valid_tgt data/valid.title.filter.txt
-save_data outputData/preprocess.output
-train_template data/train.title.txt
-valid_template data/valid.title.filter.txt

python pair_train.py 执行命令后面添加以下argument
-data outputData/preprocess.output (-gpuid 0)

python pair_rerank.py 执行命令后面添加以下arguments
-model,
-src,
-templates