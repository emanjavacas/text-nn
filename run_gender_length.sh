
MODEL=$1

# token level
for LENGTH in $(seq 10 10 100); do
	python gender.py --model $MODEL --gpu --epochs 5 --dropout 0.3 --emb_dim 300 --load_embeddings --flavor ft --suffix wiki --level token --min_len $LENGTH --concat --exp_id token_min_len
done

for LENGTH in $(seq 100 100 500); do
    python gender.py --model $MODEL --gpu --epochs 5 --dropout 0.3 --emb_dim 300 --load_embeddings --flavor ft --suffix wiki --level token --min_len $LENGTH --concat --exp_id token_min_len
done

# char level
for LENGTH in $(seq 140 200 1000); do
	python gender.py --model $MODEL --gpu --epochs 5 --dropout 0.3 --emb_dim 8 --level char --min_len $LENGTH --concat --min_freq 500 --exp_id char_min_len
done
