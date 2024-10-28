import numpy as np

# Load GloVe word vectors
def load_glove_vectors(glove_file):
    word_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

# Compute cosine similarity between two word vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Find the top 5 most similar words
def find_top_similar_words(word, word_vectors, top_n=5):
    if word not in word_vectors:
        print(f"Word '{word}' not in vocabulary.")
        return []

    word_vector = word_vectors[word]
    similarities = {}

    for other_word, other_vector in word_vectors.items():
        if other_word != word:
            similarity = cosine_similarity(word_vector, other_vector)
            similarities[other_word] = similarity

    # Sort words by cosine similarity and return the top_n
    sorted_similar_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similar_words[:top_n]

# Main
if __name__ == "__main__":
    # Load GloVe word vectors from the file
    glove_file = 'glove.6B.50d.txt'  # Path to the GloVe file
    word_vectors = load_glove_vectors(glove_file)

    # Cosine similarity between word pairs
    pairs = [("cat", "dog"), ("car", "bus"), ("apple", "banana")]
    for word1, word2 in pairs:
        if word1 in word_vectors and word2 in word_vectors:
            sim = cosine_similarity(word_vectors[word1], word_vectors[word2])
            print(f"Cosine similarity between '{word1}' and '{word2}': \n{sim:.4f}")
        else:
            print(f"One of the words '{word1}' or '{word2}' is not in vocabulary.")

    # Finding top 5 most similar words
    words_to_check = ["king", "computer", "university"]
    for word in words_to_check:
        print(f"\nTop 5 similar words to '{word}':")
        similar_words = find_top_similar_words(word, word_vectors)
        for similar_word, sim in similar_words:
            print(f"{similar_word}: {sim:.4f}")
