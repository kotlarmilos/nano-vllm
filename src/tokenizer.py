import os
import argparse
import pyarrow.parquet as pq
import json

from dataset import sample_text

ARTIFACTS_PATH = "downloads"
TOKENIZER_PATH = "tokenizer"
EOW = "<|EOW|>"

def collect_words(max_samples = 100):
    words = {}
    
    i = 0
    for s in sample_text():
        for w in s.split():
            words[w] = words.get(w, 0) + 1
            
        i += 1
        if i >= max_samples:
            break
        
    return sorted(words, key=words.get, reverse=True)
    

def train(words, max_merges = 50000):
    words = [list(w) + [EOW] for w in words]
    
    merges = []
    for _ in range(max_merges):
        tokens = {}
        for word in words:
            for i, _ in enumerate(word):
                if i < len(word) - 1:
                    pair = (word[i], word[i + 1])
                    if pair[1] == EOW:
                        continue
                    tokens[pair] = tokens.get(pair, 0) + 1
                        
        if not tokens:
            break
                    
        max_count = max(tokens.values())
        pair = [pair for pair, count in tokens.items() if count == max_count][0]
        
        matches = 0
        for word in words:
            i = 0
            while i < len(word) - 1:
                if word[i] == pair[0] and word[i+1] == pair[1]:
                    word[i:i+2] = ["".join(pair)]
                    matches += 1
                else:
                    i += 1
        
        if matches == 0:
            break

        merges.append(pair)
        
    return merges
        
def build_vocab(merges, words):
    vocab = {}
    token_id = 0
    
    base_tokens = set({c for w in words for c in w if not c.isspace()})
    for c in sorted(base_tokens):
        vocab[c] = token_id
        token_id += 1
        
    for a, b in merges:
        merged = a + b
        if merged not in vocab:
            vocab[merged] = token_id
            token_id += 1
            
    vocab[EOW] = token_id
    return vocab
        
def encode(sencence, merges, vocab):
    words = [list(word) + [EOW] for word in sencence.split(" ")]
        
    for a, b in merges:
        merged = a + b
        for word in words:
            i = 0
            while i < len(word) - 1:
                if word[i] == a and word[i + 1] == b:
                    word[i:i+2] = [merged]
                else:
                    i += 1

    tokens = []
    for word in words:
        for token in word:
            if token in vocab:
                tokens.append(vocab[token])
            else:
                raise Exception(f"Token {token} not found in the vocab")
                
    return tokens

def decode(tokens, vocab):
        inv_vocab = {v: k for k, v in vocab.items()}
        
        sentence = []
        for tok in tokens:
            key = inv_vocab[tok]
            sentence.append(" " if key == EOW else key)
            
        return "".join(sentence)

def save_tokenizer(merges, vocab, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "merges.json"), "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False)
    with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

def load_tokenizer(path):
    with open(os.path.join(path, "merges.json"), "r", encoding="utf-8") as f:
        merges = json.load(f)
    with open(os.path.join(path, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return merges, vocab

if __name__ == "__main__":
    
    words = collect_words(256)
    merges = train(words, 100)
    vocab = build_vocab(merges, words)
    print(f"Merges rules: {len(merges)}. Vocab size: {len(vocab)}")
    save_tokenizer(merges, vocab, TOKENIZER_PATH)
    print("Tokenizer saved")
    