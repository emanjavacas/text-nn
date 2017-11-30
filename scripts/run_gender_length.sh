
MODELS='DCNN RCNN CNNText ConvRec RNNText'
EPOCHS=10
ACT=relu
DROPOUT=0.3
KERNEL_SIZES='5 4 3'
OUT_CHANNELS=100
EMB_DIM=100
FLAVOR=glove
SUFFIX=twitter.27B.100d

function select_model()
{
    MODEL=$1
    case $MODEL in
	RCNN)
	    ACT=relu
	    ;;
	CNNText)
	    ACT=relu
	    ;;
	DCNN)
	    ACT=tanh
	    KERNEL_SIZES='9 7 5'
	    OUT_CHANNELS='6 10 14'
	    ;;
    esac
}

# token level
for LENGTH in $(seq 10 10 100; seq 200 100 500); do
    for MODEL in $MODELS; do
	select_model $MODEL
	python scripts/gender.py --model $MODEL --gpu --epochs $EPOCHS --dropout $DROPOUT --emb_dim $EMB_DIM --act $ACT --kernel_sizes $KERNEL_SIZES --out_channels $OUT_CHANNELS --level token --min_len $LENGTH --concat --exp_id token_min_len --cache_data --load_embeddings --flavor $FLAVOR --suffix $SUFFIX --early_stopping 2
    done
done

# char level
for LENGTH in $(seq 60 150 1000); do
    for MODEL in $MODELS; do
	select_model $MODEL
	# forcely reduce embedding dimension for character-based models
	EMB_DIM=50
	python scripts/gender.py --model $MODEL --gpu --epochs $EPOCHS --dropout $DROPOUT --emb_dim $EMB_DIM --act $ACT --kernel_size $KERNEL_SIZES --out_channels $OUT_CHANNELS --level char  --min_len $LENGTH --concat --exp_id char_min_len --cache_data --min_freq 100 --early_stopping 2
    done
done
