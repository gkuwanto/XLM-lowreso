# NMT pretraining
export NGPU=2
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name unsupMT_bgen \
--dump_path ./dumped/ \
--reload_model '/projectnb/multilm/siyang/XLM/dumped/unsupMT_bgen/hn4ahdhpsv/checkpoint.pth,/projectnb/multilm/siyang/XLM/dumped/unsupMT_bgen/hn4ahdhpsv/checkpoint.pth' \
--data_path ./data/processed/bg-en \
--lgs 'bg-en' \
--ae_steps 'bg,en' \
--bt_steps 'bg-en-bg,en-bg-en' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.1 \
--lambda_ae '0:1,100000:0.1,300000:0' \
--encoder_only false \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 2000 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 200000 \
--eval_bleu true \
--max_vocab 200000 \
--stopping_criterion 'valid_bg-en_mt_bleu,10' \
--validation_metrics 'valid_bg-en_mt_bleu'