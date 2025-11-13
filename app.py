import streamlit as st
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from gtts import gTTS
import tempfile

# ====== SET YOUR API KEY HERE ======
client = Groq(api_key="Enter your API Key....")  
# ===================================

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path="data/vectordb")

# Load embedding model
embed_model = SentenceTransformer("BAAI/bge-small-en")

st.set_page_config(page_title="📚 AI Notes Assistant", layout="wide")

# ===== Initialize session states =====
if "history" not in st.session_state:
    st.session_state.history = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

# ===== SIDEBAR NAVIGATION =====
st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.current_page = "Home"
if st.sidebar.button("📜 History"):
    st.session_state.current_page = "History"

# ===== FUNCTIONS =====
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def ask_llm(prompt):
    messages = [{"role": "system", "content": "You are a helpful study assistant."}]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages + [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def extract_keywords(text):
    prompt = f"Extract 5 keywords only, comma separated:\n\n{text}\n\nKeywords:"
    return ask_llm(prompt)

def youtube_recommendation(keywords):
    query = keywords.replace(",", "").replace("\n", " ").strip().replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}"

def text_to_audio(text):
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# ===== HOME PAGE =====
if st.session_state.current_page == "Home":
    st.title("📚 AI Notes Assistant (Home)")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        st.success("PDF uploaded ✅")
        
        # Extract PDF text only once
        st.session_state.raw_text = extract_text(uploaded_file)
        raw_text = st.session_state.raw_text

        # Split text into chunks for vector DB
        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]
        collection = chroma_client.get_or_create_collection("notes")

        # Clear old data
        old = collection.get()
        if len(old["ids"]) > 0:
            collection.delete(ids=old["ids"])

        for idx, chunk in enumerate(chunks):
            embedding = embed_model.encode(chunk).tolist()
            collection.add(documents=[chunk], embeddings=[embedding], ids=[str(idx)])

        st.success("Vector DB Ready ✅")

    # Text input for Q&A
    question = st.text_input("Ask your question:")

    # ===== BUTTONS =====
    col1, col2 = st.columns(2)
    with col1:
        answer_btn = st.button("Get Answer")
    with col2:
        summarize_btn = st.button("Summarize PDF")

    # ===== Q&A FLOW =====
    if answer_btn:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            # Try to get PDF-based answer if available
            pdf_answer = None
            if uploaded_file:
                try:
                    collection = chroma_client.get_or_create_collection("notes")
                    q_emb = embed_model.encode(question).tolist()
                    results = collection.query(query_embeddings=[q_emb], n_results=3)
                    retrieved_text = "\n".join(results["documents"][0])
                    pdf_prompt = f"Use ONLY this text to answer:\n\n{retrieved_text}\n\nQuestion: {question}\nAnswer:"
                    pdf_answer = ask_llm(pdf_prompt)

                    st.subheader("📌 Answer from PDF:")
                    st.write(pdf_answer)
                except Exception as e:
                    st.error(f"Error in PDF answer: {e}")
            else:
                st.info("No PDF uploaded — showing general answer only.")

            # Always generate general answer
            general_prompt = f"Answer this question in simple, clear language:\n\n{question}"
            general_answer = ask_llm(general_prompt)
            st.subheader("🌐 General Knowledge Answer:")
            st.write(general_answer)

            # Extract keywords and YouTube link (from PDF or general answer)
            base_text = pdf_answer if pdf_answer else general_answer
            keywords = extract_keywords(base_text)
            yt_link = youtube_recommendation(keywords)
            st.subheader("🎥 Recommended YouTube:")
            st.markdown(f"[Open YouTube Results]({yt_link})")

            # Audio
            audio_path = text_to_audio(base_text)
            st.subheader("🔊 Listen to Answer:")
            st.audio(audio_path, format="audio/mp3")

            # Save to history
            st.session_state.history.append({
                "question": question,
                "pdf_answer": pdf_answer if pdf_answer else "N/A (no PDF)",
                "youtube": yt_link
            })

    # ===== PDF SUMMARIZATION FLOW =====
    if summarize_btn:
        if not uploaded_file:
            st.warning("Upload a PDF first.")
        else:
            raw_text = st.session_state.raw_text
            summary_prompt = f"Summarize the following PDF in simple and clear points:\n\n{raw_text}\n\nSummary:"
            pdf_summary = ask_llm(summary_prompt)
            st.subheader("📝 PDF Summary:")
            st.write(pdf_summary)

            # Optional: Text-to-speech for summary
            audio_summary_path = text_to_audio(pdf_summary)
            st.subheader("🔊 Listen to PDF Summary:")
            st.audio(audio_summary_path, format="audio/mp3")

# ===== HISTORY PAGE =====
elif st.session_state.current_page == "History":
    st.title("📜 Q&A History")
    if len(st.session_state.history) == 0:
        st.info("No history yet. Ask some questions in Home page.")
    else:
        for idx, record in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{idx}: {record['question']}**")
            st.write(f"📌 PDF Answer: {record['pdf_answer']}")
            st.markdown(f"🎥 YouTube: [Link]({record['youtube']})")
            st.divider()
