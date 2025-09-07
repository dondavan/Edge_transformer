export LD_LIBRARY_PATH=lib/

d_model=768
input_len='16'
./graph_bert_base_uncased --cluster_centers=128 --d_model=$d_model --threads=4 --input_len=${input_len} --text=./data/bert-base-uncased/input/input_text_16.txt --segment=./data/bert-base-uncased/input/input_segment_16.txt --data=./data/bert-base-uncased/ --vocabulary=./data/bert-base-uncased/vocab.txt --raw-output=true
rm measure_output.txt