from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How much attendance is required?",
    "What percentage of attendance do students need?",
    "What is the hostel fee?"
]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print(sentence)
    print(embedding[:5])
    print()