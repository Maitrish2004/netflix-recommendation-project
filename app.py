from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
from difflib import get_close_matches
import os
import html
import re   # ✅ AI text parsing এর জন্য ADD

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ===============================
# User Auth storage (in-memory)
# ===============================
stored_user = {"username": None, "password": None}
watch_history = []

# ===============================
# Load dataset safely
# ===============================
DATA_PATH = os.path.join("backend", "data", "movies.csv")

movies = pd.read_csv(
    DATA_PATH,
    encoding="utf-8",
    on_bad_lines="skip"
)
movies.fillna("", inplace=True)

# ===============================
# Clean & normalize data
# ===============================
movies["title"] = movies["title"].str.strip()
movies["title_clean"] = movies["title"].str.lower().str.strip()
movies["genres"] = movies["genres"].astype(str)

def clean_video(url):
    if not isinstance(url, str):
        return ""
    url = html.unescape(url)
    url = url.replace('"', "").strip()
    if "watch?v=" in url:
        url = url.replace("watch?v=", "embed/")
    return url

movies["video_url"] = movies["video_url"].apply(clean_video)

# ===============================
# TF-IDF Recommendation Engine
# ===============================
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_movie(title, top_n=5):
    title = title.lower().strip()
    matches = movies[movies["title_clean"].str.contains(title, regex=False)]
    if matches.empty:
        close = get_close_matches(title, movies["title_clean"], n=1)
        if not close:
            raise ValueError("Movie not found")
        matches = movies[movies["title_clean"] == close[0]]
    idx = matches.index[0]
    selected = movies.loc[idx]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_movies = movies.iloc[[i[0] for i in scores]]
    return {
        "selected": {
            "title": selected["title"],
            "genres": selected["genres"],
            "video_url": selected["video_url"]
        },
        "recommended": rec_movies[["title", "genres", "video_url"]].to_dict(orient="records")
    }

# ==================================================
# ⭐ AI TEXT → GENRE DECISION SYSTEM (NEW ADD)
# ==================================================
def ai_detect_genres(user_text):
    text = user_text.lower()
    genre_map = {
        "comedy": ["funny", "comedy", "hasir", "moja"],
        "drama": ["emotional", "drama", "mon choya", "feel"],
        "action": ["action", "thrill", "fight"],
        "adventure": ["adventure", "journey"],
        "romance": ["love", "romantic"]
    }

    detected = []
    for genre, keywords in genre_map.items():
        for k in keywords:
            if re.search(r"\b" + re.escape(k) + r"\b", text):
                detected.append(genre)
                break

    return detected

# ===============================
# ⭐ AI NATURAL LANGUAGE API
# ===============================
@app.route("/ai_recommend")
def ai_recommend():
    user_text = request.args.get("text", "").strip()
    if not user_text:
        return jsonify({"status": "error", "message": "Text required"})

    genres = ai_detect_genres(user_text)
    if not genres:
        return jsonify({"status": "error", "message": "No genre detected by AI"})

    matched = movies[movies["genres"].str.lower().str.contains("|".join(genres))]
    if matched.empty:
        return jsonify({"status": "error", "message": "No movies found"})

    base_movie = matched.iloc[0]
    data = recommend_movie(base_movie["title"])

    return jsonify({
        "status": "success",
        "ai_understanding": genres,
        "data": data
    })

# ===============================
# ⭐ Genre based recommendation (UNCHANGED)
# ===============================
@app.route("/recommend_by_genre")
def recommend_by_genre():
    genre = request.args.get("genre", "").lower().strip()
    if not genre:
        return jsonify({"status": "error", "message": "Genre required"})

    matched = movies[movies["genres"].str.lower().str.contains(genre)]
    if matched.empty:
        return jsonify({"status": "error", "message": "No movies found"})

    base_movie = matched.iloc[0]
    data = recommend_movie(base_movie["title"])
    return jsonify({"status": "success", "data": data})

# ===============================
# Auth APIs (UNCHANGED)
# ===============================
@app.route("/login", methods=["POST"])
def login():
    global stored_user
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"})

    if stored_user["username"] is None:
        stored_user["username"] = username
        stored_user["password"] = password
        return jsonify({"status": "success", "message": "Login successful"})

    if username == stored_user["username"] and password == stored_user["password"]:
        return jsonify({"status": "success", "message": "Login successful"})
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid credentials",
            "forgot_allowed": True
        })

@app.route("/reset_password", methods=["POST"])
def reset_password():
    global stored_user
    data = request.get_json()
    new_username = data.get("username", "").strip()
    new_password = data.get("password", "").strip()

    if not new_username or not new_password:
        return jsonify({"status": "error", "message": "Username and Password required"})

    stored_user = {"username": new_username, "password": new_password}
    return jsonify({"status": "success", "message": "Credentials reset successful"})

# ===============================
# Movie APIs (UNCHANGED)
# ===============================
@app.route("/recommend")
def recommend_api():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"status": "error", "message": "Movie title required"})
    try:
        return jsonify({"status": "success", "data": recommend_movie(title)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/movies")
def movie_list():
    return jsonify(sorted(movies["title"].unique().tolist()))
@app.route("/add_history", methods=["POST"])
def add_history():
    global watch_history
    data = request.get_json()
    movie = { "title": data.get("title"), 
             "genres": data.get("genres"),
             "video_url":data.get("video_url") 
             }
    watch_history.insert(0,movie)
    watch_history = watch_history[:10]
    return jsonify({"status": "success"})
@app.route("/get_history")
def get_history():
    return jsonify({"status": "success", "history": watch_history})
# ==================================================
# 🧠 AI DECISION + PERCENTAGE SYSTEM (ADD ONLY)
# ==================================================

def ai_decision_intent(user_text):
    text = user_text.lower()

    mood_words = ["sad", "bad mood", "depressed", "unhappy", "lonely"]
    foolish_words = ["foolish", "stupid", "dumb", "idiot"]

    for w in mood_words:
        if w in text:
            return "mood", "I understand you are feeling low. Watching these movies can help improve your mood."

    for w in foolish_words:
        if w in text:
            return "intelligent", "You should watch intelligent movies to improve your thinking."

    return None, "Here are some movies you may like."


def generate_percentages(n):
    start = 90
    step = 10
    return [max(10, start - i * step) for i in range(n)]


@app.route("/ai_decision_recommend")
def ai_decision_recommend():
    user_text = request.args.get("text", "").strip()
    if not user_text:
        return jsonify({"status": "error", "message": "Text required"})

    intent, ai_reply = ai_decision_intent(user_text)

    # 🎭 Mood based movies
    if intent == "mood":
        matched = movies[movies["genres"].str.lower().str.contains("comedy|drama")]

    # 🧠 Intelligent movies
    elif intent == "intelligent":
        matched = movies[movies["genres"].str.lower().str.contains("drama|history|adventure")]

    else:
        return jsonify({"status": "error", "message": "AI could not decide"})

    if matched.empty:
        return jsonify({"status": "error", "message": "No movies found"})

    top_movies = matched.head(5)
    percentages = generate_percentages(len(top_movies))

    result = []
    for i, (_, row) in enumerate(top_movies.iterrows()):
        result.append({
            "title": row["title"],
            "genres": row["genres"],
            "video_url": row["video_url"],
            "percentage": percentages[i]
        })

    return jsonify({
        "status": "success",
        "intent": intent,
        "ai_reply": ai_reply,
        "movies": result
    })


# ===============================
# Serve frontend
# ===============================
@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")
if __name__ == "__main__":
    app.run(debug=True)




