import streamlit as st
from joblib import load
import re
import string 
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from googleapiclient.discovery import build
import pandas as pd
import altair as alt
from scipy.sparse import hstack # <--- NEW IMPORT NEEDED

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negation_words = {
    'no', 'not', 'never', 'none', 'neither', 'nor', "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "can't", "cannot", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't"
}
final_stop_words = stop_words - negation_words

# --- API Key ---
try:
    api_key = st.secrets["YOUTUBE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Missing Streamlit secrets. Please create .streamlit/secrets.toml")
    st.stop()


@st.cache_resource
def load_all_models():
    try:
        # Sentiment and Community models return 2 items (Classifier, Vectorizer)
        s_classifier, s_vectorizer = load('sentiment_model.joblib')
        c_classifier, c_vectorizer = load('community_model.joblib')
        
        # Constructive model now returns 3 items (Classifier, Vectorizer, Scaler)
        con_classifier, con_vectorizer, con_scaler = load('constructive_model.joblib')
        
        return (s_classifier, s_vectorizer), (c_classifier, c_vectorizer), (con_classifier, con_vectorizer, con_scaler)
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure all 3 models (.joblib files) are in the root directory and run 'models.py'.")
        st.stop()
    except ValueError as e:
        st.error(f"Model mismatch error: {e}")
        st.warning("It looks like 'constructive_model.joblib' has changed. Please re-run 'models.py' to generate the new file.")
        st.stop()

(s_classifier, s_vectorizer), (c_classifier, c_vectorizer), (con_classifier, con_vectorizer, con_scaler) = load_all_models()
st.success("All 3 models loaded successfully.")


def preprocess_standard(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in final_stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_constructive_text(text):
    if not isinstance(text, str):
        text = str(text)
    return text.lower()

def analyze_sentiment(text):
    cleaned_text = preprocess_standard(text)
    text_transformed = s_vectorizer.transform([cleaned_text])
    return s_classifier.predict(text_transformed)[0]

def analyze_community(text):
    cleaned_text = preprocess_standard(text)
    text_transformed = c_vectorizer.transform([cleaned_text])
    return c_classifier.predict(text_transformed)[0]

def analyze_constructive(text):
    cleaned_text = preprocess_constructive_text(text)
    
    text_transformed = con_vectorizer.transform([cleaned_text])
    
    text_len = len(text)
    
    len_scaled = con_scaler.transform([[text_len]])
    
    final_features = hstack([text_transformed, len_scaled])
    
    return con_classifier.predict(final_features)[0]

def generate_improvement_summary(constructive_comments):
    topic_keywords = {
        "audio": ['audio', 'mic', 'microphone', 'sound', 'loud', 'quiet', 'tinny', 'echo', 'music'],
        "visuals": ['lighting', 'light', 'dark', 'blurry', 'camera', 'video', 'quality', 'focus', 'resolution', 'color'],
        "pacing": ['long', 'short', 'pacing', 'slow', 'fast', 'boring', 'editing', 'cuts', 'duration'],
        "content": ['topic', 'subject', 'script', 'information', 'research', 'explanation', 'more', 'less']
    }
    suggestions_map = {
        "audio": "🎙️ **Audio:** Your audience mentioned the audio. Review comments about the microphone, sound levels, or background music.",
        "visuals": "💡 **Visuals:** There's feedback on the video's appearance. Check comments related to lighting, camera focus, or video quality.",
        "pacing": "⏳ **Pacing/Length:** Some viewers commented on the video's pacing. See what they're saying about the editing or duration.",
        "content": "📚 **Content:** You received feedback on the video's topic. Look for suggestions related to the script, research, or depth of information."
    }
    topics_found = set()
    suggestions_to_show = []
    for comment in constructive_comments:
        comment_lower = comment.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in comment_lower for keyword in keywords):
                if topic not in topics_found:
                    suggestions_to_show.append(suggestions_map[topic])
                    topics_found.add(topic)
    if not suggestions_to_show:
        return ["Your audience provided general constructive feedback. Read through the comments to find specific insights!"]
    return suggestions_to_show

def check_bot_heuristics(comment_text, author_name):
    flags = []
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', comment_text):
        flags.append("Contains URL")
    spam_triggers = ['check my profile', 'free crypto', 'sub to me', 'my channel']
    if any(trigger in comment_text.lower() for trigger in spam_triggers):
        flags.append("Spam Trigger Phrase")
    if re.match(r'^user-[a-zA-Z0-9]{10,}$', author_name):
        flags.append("Generic Username")
    return flags
    
@st.cache_data 
def get_youtube_comments(video_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_data = [] 
    next_page_token = None
    try:
        while len(comments_data) < max_results:
            request = youtube.commentThreads().list(
                part="snippet", videoId=video_id,
                maxResults=min(100, max_results - len(comments_data)),
                textFormat="plainText", pageToken=next_page_token
            )
            response = request.execute()
            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comment_text = snippet['textDisplay']
                author_name = snippet['authorDisplayName']
                comments_data.append((comment_text, author_name))
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        st.warning("Comments might be disabled for this video.")
        return []
    return comments_data
    
def extract_video_id(url):
    patterns = [ r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)" ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# --- UI ---
st.title("YouTube Comment Analyzer 3.0")
st.markdown("Analyze **Sentiment**, **Community Health**, and **Constructive Feedback**.")

if 'page' not in st.session_state:
    st.session_state.page = "Sentiment Analysis"
if 'df' not in st.session_state:
    st.session_state.df = None

url = st.text_input("Enter Youtube Video URL:")
analyze_button = st.button("Analyze Comments")

if analyze_button and url:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
    else:
        with st.spinner("Fetching Comments..."):
            comments_data = get_youtube_comments(video_id)
            st.write(f"Video ID: {video_id}, Fetched {len(comments_data)} comments.")
        
        if not comments_data:
            st.warning("No comments found for this video.")
            st.session_state.df = None # Clear old results
        else:
            results = []
            with st.spinner("Analyzing comments with all 3 models..."):
                for comment_text, author_name in comments_data:
                    sentiment = analyze_sentiment(comment_text)
                    community_label = analyze_community(comment_text)
                    constructive_label = analyze_constructive(comment_text)
                    bot_flags = check_bot_heuristics(comment_text, author_name)
                    
                    final_health_label = community_label
                    if community_label != 'spam' and bot_flags:
                        final_health_label = 'spam' 
                    
                    results.append({
                        "Comment": comment_text,
                        "Author": author_name,
                        "Sentiment": sentiment,
                        "Health": final_health_label,
                        "Feedback": constructive_label,
                        "Flags": ", ".join(bot_flags) if bot_flags else ""
                    })
            
            st.session_state.df = pd.DataFrame(results)
            st.session_state.page = "Sentiment Analysis" 
            st.rerun() 

if st.session_state.df is not None:
    df = st.session_state.df

    st.sidebar.title("Navigation")
    st.sidebar.radio(
        "Choose a section to view:",
        ["Sentiment Analysis", "Community Health", "Constructive Feedback", "All Data"],
        key='page' 
    )
    st.sidebar.markdown("---")


    if st.session_state.page == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment_counts_series = df["Sentiment"].value_counts()
        
        st.bar_chart(sentiment_counts_series) 
        
        positive_count = sentiment_counts_series.get('Positive', 0)
        negative_count = sentiment_counts_series.get('Negative', 0)
        neutral_count = sentiment_counts_series.get('Neutral', 0)
        irrelevant_count = sentiment_counts_series.get('Irrelevant', 0)
        
        try:
            overall_sentiment = sentiment_counts_series.idxmax()
        except ValueError:
            overall_sentiment = "N/A"

        st.metric("Overall Sentiment", value=overall_sentiment)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive", positive_count)
        col2.metric("Negative", negative_count)
        col3.metric("Neutral", neutral_count)
        col4.metric("Irrelevant", irrelevant_count)

    elif st.session_state.page == "Community Health":
        st.subheader("Community Health Analysis")
        health_counts_series = df["Health"].value_counts()
        health_counts_df = health_counts_series.reset_index()
        health_counts_df.columns = ['Health', 'Count']

        pie_health = alt.Chart(health_counts_df).mark_arc(outerRadius=120).encode(
            theta=alt.Theta("Count", stack=True),
            color=alt.Color("Health", scale={'domain': ['healthy', 'toxic', 'spam'],
                                             'range': ['#2ca02c', '#d62728', '#ff7f0e']}),
            tooltip=["Health", "Count", alt.Tooltip("Count", format=".1%")]
        ).properties(title="Community Health")
        st.altair_chart(pie_health, use_container_width=True) 

        healthy_count = health_counts_series.get('healthy', 0)
        toxic_count = health_counts_series.get('toxic', 0)
        spam_count = health_counts_series.get('spam', 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Healthy", healthy_count)
        col2.metric("☠️ Toxic", toxic_count)
        col3.metric("🤖 Spam/Bot", spam_count)
        
        st.subheader("Flagged for Moderation (Toxic or Spam)")
        flagged_df = df[(df['Health'] == 'toxic') | (df['Health'] == 'spam')]
        st.dataframe(flagged_df[['Author', 'Comment', 'Health', 'Flags']])

    elif st.session_state.page == "Constructive Feedback":
        st.subheader("Constructive Feedback Analysis")
        constructive_counts_series = df["Feedback"].value_counts()
        constructive_counts_df = constructive_counts_series.reset_index()
        constructive_counts_df.columns = ['Feedback', 'Count']

        pie_constructive = alt.Chart(constructive_counts_df).mark_arc(outerRadius=120).encode(
            theta=alt.Theta("Count", stack=True), 
            color=alt.Color("Feedback"),
            tooltip=["Feedback", "Count", alt.Tooltip("Count", format=".1%")]
        ).properties(title="Feedback Type")
        st.altair_chart(pie_constructive, use_container_width=True)
        
        con_count = constructive_counts_series.get('Constructive', 0)
        non_con_count = constructive_counts_series.get('Non-Constructive', 0)

        col1, col2 = st.columns(2)
        col1.metric("✅ Constructive", con_count)
        col2.metric("❌ Non-Constructive", non_con_count)
        
        st.markdown("---") 

        st.subheader("Actionable Improvement Guide")
        constructive_df = df[df['Feedback'] == 'Constructive']
        
        if not constructive_df.empty:
            constructive_list = constructive_df['Comment'].tolist()
            suggestions = generate_improvement_summary(constructive_list)
            
            st.markdown("Based on your constructive comments, here are the key areas to focus on:")
            for suggestion in suggestions:
                st.info(suggestion) 
        else:
            st.info("No constructive comments were found to generate an improvement guide.")
        
        st.markdown("---") 

        st.subheader("All Constructive Comments")
        st.dataframe(constructive_df[['Author', 'Comment']])

    elif st.session_state.page == "All Data":
        st.subheader("All Analyzed Data")
        st.dataframe(df)