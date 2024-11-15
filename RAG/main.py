from openai import OpenAI
import numpy as np

client = OpenAI(api_key="")

# Load file, create embeddings for 50-word chunks
with open("messi.txt", "r") as file:
    content = file.read().split()
    
embeddings_dict = {} #dict

# split the txt file into sections
numSections = 50

wordsPer = len(content)//numSections
if len(content) % numSections: # round up if not /100 equals whole number
    wordsPer += 1

for pos in range(numSections):
    section = content[wordsPer*pos :wordsPer*(pos+1)]
    sentence = " ".join(section)
    
    embeddings_dict[sentence] = client.embeddings.create(
        input=sentence,
        model="text-embedding-ada-002"
    ).data[0].embedding


# Get user prompt and compute embedding
prompt = input("Ask a question about Messi: ")
query_embedding = client.embeddings.create(
    input=prompt,
    model="text-embedding-ada-002"
).data[0].embedding

# Calculate cosine similarity and find top 5
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# Store top 5 most similar chunks
top_five = []

for key, embedding in embeddings_dict.items():
    similarity = cosine_similarity(np.array(query_embedding), np.array(embedding))
    if len(top_five) < 5:
        top_five.append((similarity, key))
    else:
        # Replace the minimum similarity if current similarity is higher
        min_similarity = min(top_five, key=lambda x: x[0])
        if similarity > min_similarity[0]:
            top_five.remove(min_similarity)
            top_five.append((similarity, key))

# Sort top_five by similarity
top_five = sorted(top_five, key=lambda x: x[0], reverse=True)

# Final output
output = "\n".join([sentence for _, sentence in top_five])
print("\nTop 5 relevant passages:\n", output.strip())
