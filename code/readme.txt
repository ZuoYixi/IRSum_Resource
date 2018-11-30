The update of opennmt overwrited a part of our code and made the model unable to run. We are trying to fix these bugs.

python pair_preprocess.py -train_src data/train.article.txt -train_tgt ../result -valid_src data/valid.article.filter.txt -valid_tgt ../result -save_data ../result -train_template data/train.title.txt -valid_template data/valid.title.filter.txt
