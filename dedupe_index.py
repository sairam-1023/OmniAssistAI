"""One-off cleanup: removes duplicate chunks (same filename) from the FAISS index, keeping the first occurrence of each."""

import json
import pickle

import faiss
import numpy as np

INDEX_DIR = "models/language/vector_index"

with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open(f"{INDEX_DIR}/metadata.json") as f:
    metadata = json.load(f)

old_index = faiss.read_index(f"{INDEX_DIR}/index.faiss")

seen_filenames = set()
keep_indices = []
for i, m in enumerate(metadata):
    if m["filename"] not in seen_filenames:
        seen_filenames.add(m["filename"])
        keep_indices.append(i)

print(f"Original: {len(metadata)} chunks. Keeping {len(keep_indices)} (removing {len(metadata) - len(keep_indices)} duplicates).")

new_chunks = [chunks[i] for i in keep_indices]
new_metadata = [metadata[i] for i in keep_indices]

# Rebuild the FAISS index from scratch with only the deduplicated vectors.
all_vectors = old_index.reconstruct_n(0, old_index.ntotal)
new_vectors = np.array([all_vectors[i] for i in keep_indices]).astype("float32")

new_index = faiss.IndexFlatL2(old_index.d)
new_index.add(new_vectors)

faiss.write_index(new_index, f"{INDEX_DIR}/index.faiss")
with open(f"{INDEX_DIR}/chunks.pkl", "wb") as f:
    pickle.dump(new_chunks, f)
with open(f"{INDEX_DIR}/metadata.json", "w") as f:
    json.dump(new_metadata, f, indent=2)

print("Deduplication complete.")