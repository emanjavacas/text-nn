
# token level
for LENGTH in $(seq 10 10 100); do
    if [ $LENGTH -eq 10 ]; then
	python gender.py --gpu --epochs 5 --dropout 0.3 --emb_dim 300 --load_embeddings --flavor ft --suffix wiki --level token --cache_data --min_len $LENGTH --concat
    else
	python gender.py --gpu --epochs 5 --dropout 0.3 --emb_dim 300 --load_embeddings --flavor ft --suffix wiki --level token --load_data --min_len $LENGTH --concat
    fi
done

for LENGTH in $(seq 100 100 500); do
    python gender.py --gpu --epochs 5 --dropout 0.3 --emb_dim 300 --load_embeddings --flavor ft --suffix wiki --level token --load_data --min_len $LENGTH --concat
done

# char level
for LENGTH in $(seq 140 200 1000); do
    if [ $LENGTH -eq 140 ]; then
	python gender.py --gpu --epochs 5 --dropout 0.3 --emb_dim 24 --level char --cache_data --min_len $LENGTH --concat --min_freq 500
    else
	python gender.py --gpu --epochs 5 --dropout 0.3 --emb_dim 24 --level char --load_data --min_len $LENGTH --concat --min_freq 500
    fi
done
