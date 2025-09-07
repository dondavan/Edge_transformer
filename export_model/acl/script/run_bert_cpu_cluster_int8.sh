export LD_LIBRARY_PATH=lib/

d_model=768
input_len='16'
./graph_bert_base_uncased --tensor_data_type=s8 --d_model=$d_model --threads=4 --input_len=${input_len} --text=./data/bert-clustered-labelled-int8/input/input_text_16.txt --segment=./data/bert-clustered-labelled-int8/input/input_segment_16.txt --data=./data/bert-clustered-labelled-int8/ --vocabulary=./data/bert-clustered-labelled-int8/vocab.txt --raw-output=true --cluster_centers=128
rm measure_output.txt