from pathlib import Path
import json
import random
import nltk
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
 
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
 
random.seed(42)
 
TEXT_DIR = Path("../texts")
txt_paths = sorted(TEXT_DIR.glob("*.txt"))
 
TARGET_WORDS = 120
MIN_WORDS = 5
MAX_WORDS = 200
 
# ── Step 1: Load & inspect ───────────────────────────────────────────────────
files = sorted(TEXT_DIR.glob("*.txt"))
print(f"Found {len(files)} .txt files.")
 
example_file = files[0]
with open(example_file, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
print("Reading file:", example_file)
print("Number of characters:", len(text))
print("\nFirst 1,000 characters:\n")
print(text[:1000])
 
# ── Step 2: Segment & chunk (single file check) ──────────────────────────────
sample_path = txt_paths[0]
with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
 
sentences = nltk.sent_tokenize(text)
chunks = []
current = []
current_len = 0
 
for sent in sentences:
    words = sent.split()
    if not words:
        continue
    if current_len + len(words) > TARGET_WORDS and current:
        chunks.append(" ".join(current))
        current = []
        current_len = 0
    current.append(sent)
    current_len += len(words)
if current:
    chunks.append(" ".join(current))
 
lengths = [len(c.split()) for c in chunks]
lengths_sorted = sorted(lengths)
print("Number of chunks:", len(chunks))
print("  min:", min(lengths))
print("  median:", lengths_sorted[len(lengths_sorted)//2])
print("  max:", max(lengths))
lo, hi = 5, 200
in_range = sum(lo <= n <= hi for n in lengths)
print(f"Chunks with {lo}-{hi} words:", in_range)
print("Share in range:", round(in_range / len(lengths), 3))
print("\n--- Chunk 1 preview ---")
print(chunks[0][:400])
 
# ── Step 3: Tokenize (single file check) ─────────────────────────────────────
token_lists_sample = [simple_preprocess(c, deacc=True) for c in chunks]
print("\nChunks (strings):", len(chunks))
print("Chunks (token lists):", len(token_lists_sample))
print("Token preview:", token_lists_sample[0][:60])
print("Token count of first chunk:", len(token_lists_sample[0]))
 
# ── Shared chunk_text function ───────────────────────────────────────────────
def chunk_text(text, target_words=120):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if current_len + len(words) > target_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks
 
# ── Step 4: Process all files ────────────────────────────────────────────────
all_chunks = []
all_token_lists = []
print(f"\nFound {len(txt_paths)} text files.")
 
for path in txt_paths:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for c in chunk_text(text, TARGET_WORDS):
        tokens = simple_preprocess(c, deacc=True)
        if MIN_WORDS <= len(tokens) <= MAX_WORDS:
            all_chunks.append(c)
            all_token_lists.append(tokens)
 
print("Total chunks kept (after filtering):", len(all_chunks))
 
# ── Step 5: Train Word2Vec ───────────────────────────────────────────────────
print("\nTraining Word2Vec...")
model = Word2Vec(
    sentences=all_token_lists,
    vector_size=200,
    window=5,
    min_count=5,
    workers=4,
    sg=1
)
Path("../models").mkdir(exist_ok=True)
model_path = Path("../models") / "w2v_full.bin"
model.save(str(model_path))
print("Model saved to:", model_path)
 
# ── Step 5b: Query Word2Vec ──────────────────────────────────────────────────
seed = "merchant"
if seed not in model.wv:
    print(f"'{seed}' not found in vocabulary.")
else:
    print(f"\nTop 30 words similar to '{seed}':")
    for word, score in model.wv.similar_by_word(seed, topn=30):
        print(f"  {word:20s} {score:.3f}")
 
# ── Step 6: Define tiers ─────────────────────────────────────────────────────
TIER_A = {"merchant", "merchants", "marchant", "marchants"}
TIER_B = {"factor", "chapman", "adventurer", "adventurers",
          "venturer", "venturers", "staple", "staplers", "trade", "purser"}
TIER_C = {"clothier", "clothyer", "tailor", "tayler", "haberdasher",
          "goldsmith", "vintner", "brewer", "banker", "grazier", "jeweller"}
 
print("\nTier A:", sorted(TIER_A))
print("Tier B:", sorted(TIER_B))
print("Tier C:", sorted(TIER_C))
 
# ── Step 7: Label chunks ─────────────────────────────────────────────────────
labeled = []
print(f"\nProcessing {len(txt_paths)} files for labeling...")
 
for path in txt_paths:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for c in chunk_text(text, TARGET_WORDS):
        tokens = simple_preprocess(c, deacc=True)
        if not (MIN_WORDS <= len(tokens) <= MAX_WORDS):
            continue
        token_set = set(tokens)
        if token_set & (TIER_A | TIER_B):
            label = 1
        elif token_set & TIER_C:
            label = 2
        else:
            label = 0
        labeled.append((c, label))
 
print("Total chunks labeled:", len(labeled))
 
Path("../data").mkdir(exist_ok=True)
with open(Path("../data") / "merchant_labeled_chunks.json", "w", encoding="utf-8") as f:
    json.dump(labeled, f, ensure_ascii=False)
print("Saved to ../data/merchant_labeled_chunks.json")
 
print("\nLabel distribution:")
print(f"  CORE (1): {sum(1 for _, y in labeled if y == 1)}")
print(f"  NEG  (0): {sum(1 for _, y in labeled if y == 0)}")
print(f"  MAYBE(2): {sum(1 for _, y in labeled if y == 2)}")
 
for label_name, label_val in [("CORE", 1), ("MAYBE", 2), ("NEG", 0)]:
    example = next((t for t, y in labeled if y == label_val), None)
    if example:
        print(f"\n{label_name} example (first 200 chars):")
        print(example[:200])
 
# ── Step 9: Prepare datasets ─────────────────────────────────────────────────
core  = [(t, 1) for (t, y) in labeled if y == 1]
neg   = [(t, 0) for (t, y) in labeled if y == 0]
maybe = [t for (t, y) in labeled if y == 2]
 
print("\nLoaded:")
print("  CORE:", len(core))
print("  NEG :", len(neg))
print("  MAYBE:", len(maybe))
 
neg_sample = random.sample(neg, len(core))
training_data = core + neg_sample
random.shuffle(training_data)
 
print("Training set size (CORE + NEG):", len(training_data))
 
split = int(0.8 * len(training_data))
train_data = training_data[:split]
test_data  = training_data[split:]
 
print("Train size:", len(train_data))
print("Test size :", len(test_data))
print("MAYBE size:", len(maybe))
 
with open(Path("../data") / "train_core_vs_neg.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False)
with open(Path("../data") / "test_core_vs_neg.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False)
with open(Path("../data") / "maybe_texts.json", "w", encoding="utf-8") as f:
    json.dump(maybe, f, ensure_ascii=False)
 