import os
import re
import numpy as np
from tqdm import tqdm
import spacy


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\([^()]*\)', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def load_rawdata(fname):
    print("Loading Normal Text...")
    normal_text = []
    with open(os.path.join(fname, "normal.aligned")) as f:
        lines = f.readlines()
        for line in lines:
            normal_text.append(clean_text(line.split("\t")[2]))

    print("Loading Simplified Text...")
    simple_text = []
    with open(os.path.join(fname, "simple.aligned")) as f:
        lines = f.readlines()
        for line in lines:
            simple_text.append(clean_text(line.split("\t")[2]))

    return normal_text, simple_text


def vectorize(X):
    word2vec = spacy.load('en_core_web_md')
    vectorized_X = []
    for x in tqdm(X):
        vectorized_X.append(word2vec(x).vector)
    return vectorized_X


def load_dataset(fname):
    return np.load(fname, allow_pickle=True)


normal, simple = load_rawdata("./datasets/wiki-dataset")
print("Converting Dataset to Vector...")
X = vectorize(normal)
np.save('./datasets/normal.npy', X, allow_pickle=True)
Y = vectorize(simple)
np.save('./datasets/simple.npy', Y, allow_pickle=True)