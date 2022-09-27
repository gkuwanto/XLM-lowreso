# Unsupervised NMT pretraining using multiple gpus
# Note you need to change reload_model, data_path, tokens_per_batch (3000 might not fit into every memory size) and batch_size (again depending on the memory)
# Note that if you provide tokens_per_batch, it's used for training not batch_size. In this case, batch_size is only used for evalution.
# You may not need a large epoch_size when developing the model. Set a small number while debugging like 6000 so you reach the evaluation quickly to see if you can evaluate without errors. Then change back to a large number for training.
export NGPU=2
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name engu_mlm_bt_mt_cs \
--dump_path ./dumped/ \
--reload_model '/projectnb/multilm/gkuwanto/NMT/XLM/dumped/lm_gu_cs_guppl_constrained/19571/best-valid_gu_mlm_ppl.pth,/projectnb/multilm/gkuwanto/NMT/XLM/dumped/lm_gu_cs_guppl_constrained/19571/best-valid_gu_mlm_ppl.pth' \
--data_path ./data/not_rom/code-switch/new/orig/gu/processed \
--lgs 'en-gu-cs' \
--bt_steps 'en-gu-en,gu-en-gu' \
--mt_steps "en-gu,gu-en" \
--encoder_only false \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 1000 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 100000 \
--eval_bleu true \
--max_vocab 200000 \
--stopping_criterion 'valid_en-gu_mt_bleu,10' \
--validation_metrics 'valid_en-gu_mt_bleu'
