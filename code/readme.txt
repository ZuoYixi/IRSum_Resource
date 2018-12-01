The update of opennmt overwrited a part of our code and made the model unable to run. We are trying to fix these bugs.

python pair_preprocess.py -train_src data/train.article.txt -train_tgt data/train.title.txt -valid_src data/valid.article.filter.txt -valid_tgt data/valid.title.filter.txt -save_data outputData/preprocess.output -train_template data/train.title.txt -valid_template data/valid.title.filter.txt
