# LM Pretraining
export NGPU=2 
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name test_engu_mlm_tlm_with_para \
--dump_path ./dumped/ \
--data_path ./data/en-gu/processed/en-gu/ \
--lgs "en-gu" \
--clm_steps "" \
--mlm_steps "en,gu,en-gu" \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--batch_size 32 \
--bptt 256 \
--optimizer "adam,lr=0.0001" \
--epoch_size 200000 \
--validation_metrics "_valid_mlm_ppl" \
--stopping_criterion "_valid_mlm_ppl,10"