import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Convert genres into vectors
cv = CountVectorizer()
genre_matrix = cv.fit_transform(df['genre'])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

# Recommendation function
def recommend(movie_name):
    if movie_name not in df['title'].values:
        print("❌ Movie not found!")
        return

    index = df[df['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Recommended movies similar to '{movie_name}':\n")
    for i in distances[1:6]:
        print(df.iloc[i[0]].title)

# User input
while True:
    movie = input("\nEnter movie name (or type exit): ")
    if movie.lower() == "exit":
        break
    recommend(movie)
