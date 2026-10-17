
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Read college data
with open("college_data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Create chunks
sections = [
    section.strip()
    for section in text.split("\n\n")
    if section.strip()
]

chunks = []

for i in range(0, len(sections) - 1, 2):
    heading = sections[i]
    content = sections[i + 1]
    chunks.append(heading + "\n\n" + content)

# 3. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)
embeddings = np.asarray(embeddings, dtype="float32")

# 4. Build FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Create storage folder
os.makedirs("vector_store", exist_ok=True)

# 6. Save FAISS index
faiss.write_index(index, "vector_store/college.index")

# 7. Save original text chunks
with open("vector_store/chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4, ensure_ascii=False)

print("FAISS index saved successfully!")
print("Total chunks:", len(chunks))
print("Total vectors:", index.ntotal)
