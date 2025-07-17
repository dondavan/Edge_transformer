export LD_LIBRARY_PATH=lib/

d_model=768
input_len='7'
./graph_bert_base_uncased --d_model=$d_model --threads=4 --input_len=${input_len} --text=./data/bert-clustered-centered/input/input_text.txt --segment=./data/bert-clustered-centered/input/input_segment.txt --data=./data/bert-clustered-centered/ --vocabulary=./data/bert-clustered-centered/vocab.txt --raw-output=true
rm measure_output.txt