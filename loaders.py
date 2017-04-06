
import os
import json
from dataset import Dict, PairedDataset


# Data loaders
def load_twisty(path='/home/corpora/TwiSty/twisty-EN'):
    """
    Load twisty dataset with gender labels per tweet
    """
    tweets_path = os.path.join(path, 'data/tweets/en/users_id/')
    tweet_fs = set(os.listdir(tweets_path))
    with open(os.path.join(path, 'TwiSty-EN.json'), 'r') as fp:
        metadata = json.load(fp)
    src, trg = [], []
    for user_id, user_metadata in list(metadata.items()):
        if user_id + ".json" in tweet_fs:
            with open(os.path.join(tweets_path, user_id + '.json'), 'r') as fp:
                user_data = json.load(fp)
            for tweet in user_metadata['confirmed_tweet_ids']:
                src.append(user_data['tweets'][str(tweet)]['text'].split())
                trg.append([user_metadata["gender"]])
    return src, trg


def sort_key(pair):
    """
    Sort examples by tweet length
    """
    src, trg = pair
    return len(src)


def load_dataset(src, trg, batch_size, max_size=20000, min_freq=5,
                 gpu=False, shuffle=True, sort_key=sort_key, **kwargs):
    """
    Wrapper function for dataset with sensible, overwritable defaults
    """
    tweets_dict = Dict(pad_token='<pad>', eos_token='<eos>',
                       bos_token='<bos>', max_size=max_size, min_freq=min_freq)
    labels_dict = Dict(sequential=False, force_unk=False)
    tweets_dict.fit(src)
    labels_dict.fit(trg)
    d = {'src': tweets_dict, 'trg': labels_dict}
    splits = PairedDataset(src, trg, d, batch_size, gpu=gpu).splits(
        shuffle=shuffle, sort_key=sort_key, **kwargs)
    return splits
