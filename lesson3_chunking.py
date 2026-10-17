with open("college_data.txt", "r", encoding="utf-8") as file:
    text = file.read()

sections = text.split("\n\n")

chunks = []

for i in range(0, len(sections), 2):
    heading = sections[i]
    content = sections[i + 1]

    chunk = heading + "\n\n" + content
    chunks.append(chunk)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)