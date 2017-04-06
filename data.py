
import os
import json
from dataset import Dict, PairedDataset


def load_twisty(path='/home/corpora/TwiSty/twisty-EN',
                max_size=20000, labels=('gender', 'mbti')):
    tweets_path = os.path.join(path, 'data/tweets/en/users_id/')
    tweet_fs = set(os.listdir(tweets_path))
    with open(os.path.join(path, 'TwiSty-EN.json'), 'r') as fp:
        metadata = json.load(fp)
    src, trg = [], []
    for user_id, user_metadata in metadata.items():
        if user_id + ".json" in tweet_fs:
            with open(os.path.join(tweets_path, user_id + '.json'), 'r') as fp:
                user_data = json.load(fp)
            for tweet in user_metadata['confirmed_tweet_ids']:
                src.append(user_data['tweets'][str(tweet)]['text'])
                trg.append(user_metadata["gender"])
    tweets_dict = Dict(pad_token='<pad>', eos_token='<eos>',
                       bos_token='<bos>', max_size=max_size)
    labels_dict = Dict(sequential=False)
    tweets_dict.fit(src), labels_dict.fit(trg)
    return src, trg, {'src': tweets_dict, 'trg': labels_dict}
