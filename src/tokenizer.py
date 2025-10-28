import os
import argparse
import pyarrow.parquet as pq
import json
import re

ARTIFACTS_PATH = "downloads"
TOKENIZER_PATH = "tokenizer"
TOKENS = {
    "space": "<|space|>"
}

def train(text: str, max_merges = 50000):
    words = [list(word) + [TOKENS["space"]] for word in text.split(" ")]
    
    merges = []
    while True:
        tokens = {}
        for word in words:
            for i, _ in enumerate(word):
                if i < len(word) - 1:
                    pair = (word[i], word[i + 1])
                    if pair[1] == TOKENS["space"]:
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
        
        if len(merges) >= max_merges:
            break
        
    return merges
        
def build_vocab(merges, text):
    vocab = {}
    token_id = 0
    
    base_tokens = set(c for c in text if not c.isspace())
    for c in sorted(base_tokens):
        vocab[c] = token_id
        token_id += 1
        
    for a, b in merges:
        merged = a + b
        if merged not in vocab:
            vocab[merged] = token_id
            token_id += 1
            
    vocab[TOKENS["space"]] = token_id
    return vocab
        
def encode(text, merges, vocab):
    words = [list(word) + [TOKENS["space"]] for word in text.split(" ")]
        
    for a, b in merges:
        merged = a + b
        for word in words:
            i = 0
            while i < len(word) - 1:
                if word[i] == a and word[i + 1] == b:
                    word[i:i+2] = [merged]
                else:
                    i += 1

    ids = []
    for word in words:
        for token in word:
            if token in vocab:
                ids.append(vocab[token])
            else:
                raise Exception(f"Token {token} not found in the vocab")
                
    return ids

def decode(tokens, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    
    out = []
    for tok in tokens:
        key = inv_vocab[tok]
        out.append(" " if key == TOKENS["space"] else key)
    return "".join(out)

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

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
    parser = argparse.ArgumentParser(description="Simple BPE tokenizer")
    parser.add_argument("-n", "--num-shards", type=int, default=256, help="Number of shards to train on (default: 256)")
    args = parser.parse_args()
    
    texts_buf = []

    if os.path.exists(ARTIFACTS_PATH):
        files = os.listdir(ARTIFACTS_PATH)
        i = 0
        for file in files:
            pf = pq.ParquetFile(f"{ARTIFACTS_PATH}/{file}")
            for batch in pf.iter_batches(columns=["text"], batch_size=256):
                if i >= args.num_shards / 256:
                    break
                texts_buf.extend(batch.column(0).to_pylist())
                i += 1
                
    print(f"Number of sentences: {len(texts_buf)}")
    global_text = normalize(" ".join(texts_buf))
    print("Traning BPE tokenizer...")
    merges = train(global_text, 100)
    vocab = build_vocab(merges, global_text)
    print(f"Merges rules: {len(merges)}. Vocab size: {len(vocab)}")

    save_tokenizer(merges, vocab, TOKENIZER_PATH)
    
    print(f"Tokenizer saved")