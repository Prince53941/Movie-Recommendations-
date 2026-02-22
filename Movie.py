import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df["genres"] = df["genres"].fillna("")
    return df

movies = load_data()
movie_titles = movies["title"].tolist()

# ---------------- TF-IDF ---------------- #

@st.cache_resource
def tfidf_model(data):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(data["genres"])
    return tfidf, matrix

tfidf, tfidf_matrix = tfidf_model(movies)

# ---------------- MOVIE BASED RECOMMEND ---------------- #

def recommend_by_movie(user_movie, top_n=5):

    matches = movies[movies["title"].str.contains(user_movie, case=False, na=False)]

    if matches.empty:
        close = get_close_matches(user_movie, movie_titles, n=5, cutoff=0.6)
        return None, close

    idx = matches.index[0]

    input_movie = movies.loc[idx, "title"]
    input_genres = movies.loc[idx, "genres"]

    cosine_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = cosine_scores.argsort()[::-1][1:top_n+1]

    results = []

    for i in sim_indices:
        results.append({
            "Movie": movies.loc[i, "title"],
            "Genre": movies.loc[i, "genres"],
            "Similarity (%)": round(float(cosine_scores[i])*100,2)
        })

    return {
        "Input Movie": input_movie,
        "Genre": input_genres,
        "Results": results
    }, None

# ---------------- GENRE BASED RECOMMEND ---------------- #

def recommend_by_genre(user_genre, top_n=5):

    filtered = movies[movies["genres"].str.lower().str.contains(user_genre.lower())]

    if filtered.empty:
        return None

    return filtered.head(top_n)[["title","genres"]]

# ---------------- UI ---------------- #

option = st.radio("Choose Recommendation Type:", ["By Movie Name","By Genre"])

if option == "By Movie Name":

    movie = st.text_input("Enter Movie Name")

    if st.button("Recommend"):

        if movie:

            result, suggestions = recommend_by_movie(movie)

            if result:

                st.subheader("Input Movie")
                st.write(result["Input Movie"])
                st.write("Genre:", result["Genre"])

                st.subheader("Recommended Movies")

                for r in result["Results"]:
                    st.markdown(f"""
                    **🎬 {r['Movie']}**  
                    Genre: {r['Genre']}  
                    Similarity: {r['Similarity (%)']}%
                    ---
                    """)

            else:
                st.error("Movie not found.")
                if suggestions:
                    st.info("Did you mean:")
                    for s in suggestions:
                        st.write("-", s)

        else:
            st.warning("Please enter a movie name.")

else:

    genre = st.text_input("Enter Genre (example: drama, horror)")

    if st.button("Recommend"):

        if genre:

            movies_by_genre = recommend_by_genre(genre)

            if movies_by_genre is not None:
                st.subheader("Top Movies")
                st.dataframe(movies_by_genre)

            else:
                st.error("No movies found for this genre.")

        else:
            st.warning("Please enter genre.")
