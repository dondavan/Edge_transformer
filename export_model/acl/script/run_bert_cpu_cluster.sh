export LD_LIBRARY_PATH=lib/

d_model=768
input_len='7'
./graph_bert_base_uncased --d_model=$d_model --threads=4 --input_len=${input_len} --text=./data/bert-clustered-labelled/input/input_text.txt --segment=./data/bert-clustered-labelled/input/input_segment.txt --data=./data/bert-clustered-labelled/ --vocabulary=./data/bert-clustered-labelled/vocab.txt --raw-output=true --cluster_centers=128
rm measure_output.txt