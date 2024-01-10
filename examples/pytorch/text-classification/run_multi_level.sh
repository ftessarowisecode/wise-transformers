# data_dir=/Users/filippotessaro/Desktop/wrk/WISEcode/product-categorization/data/processed
data_dir=/Users/filippotessaro/Desktop/wrk/WISEcode/product-categorization/data/processed
out_dir=./tmp_model

# python sklearn_classifier.py \
#     --train_file $data_dir/train.csv \
#     --test_file $data_dir/test.csv \
#     --label_column_name cat2 \
#     --text_column_names norm_name \
#     --sep "\t" \
#     --output_dir $out_dir/sklearn_exp


python examples/pytorch/text-classification/run_multi_level_classification.py \
    --train_file $data_dir/train.csv \
    --validation_file $data_dir/val.csv \
    --test_file $data_dir/test.csv \
    --do_train true \
    --do_eval true \
    --do_predict true \
    --pad_to_max_length true \
    --max_seq_length 256 \
    --label_column_name_l1 cat1 \
    --label_column_name_l2 cat2 \
    --text_column_names norm_name \
    --shuffle_seed 42 \
    --shuffle_train_dataset true \
    --model_name_or_path distilbert-base-uncased \
    --use_fast_tokenizer True \
    --output_dir  $out_dir \
    --epochs 5 \
    --overwrite_output_dir \
    --max_train_samples 100 \
    --max_eval_samples 100 \
    --max_predict_samples 100 \
    # --add_extra_tokens 50

data_dir=/Users/filippotessaro/Desktop/wrk/WISEcode/product-categorization/data/processed
# model_path=/Users/filippotessaro/Desktop/models_to_be_tested/exp_distilbert-base-uncased_pino_data_clean_2jan_L2_ep10
model_path=/Users/filippotessaro/Downloads/best_weights_hf/
out_dir=./tmp_L2_ligth

echo run eval 
# run eval
# python transformers_hf_classification_trainer.py \
#     --train_file $data_dir/train.csv \
#     --validation_file $data_dir/val.csv \
#     --test_file $data_dir/test.csv \
#     --do_predict true \
#     --pad_to_max_length true \
#     --max_seq_length 256 \
#     --label_column_name cat2 \
#     --text_column_names norm_name \
#     --shuffle_seed 42 \
#     --shuffle_train_dataset true \
#     --model_name_or_path $model_path \
#     --use_fast_tokenizer True \
#     --output_dir  $out_dir \
#     --overwrite_output_dir 
    # --max_predict_samples 100 \