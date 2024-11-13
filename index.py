import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
from keybert import KeyBERT
from io import StringIO


def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def read_word(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


st.markdown(
    """
    <h1 style='text-align: center; color: #2A9D8F;'>Text Summarization, Keyword Identification, and Title Generation</h1>
    <hr style='border: 1px solid #264653;'>
    """, 
    unsafe_allow_html=True
)


def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-base", truncation=True)
    keyword_model = KeyBERT()
    title_generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return summarizer, keyword_model, title_generator


summarizer, keyword_model, title_generator = load_models()


st.sidebar.markdown(
    """
    <h2 style='text-align: center; color: #E76F51;'>Options</h2>
    """, 
    unsafe_allow_html=True
)


generate_keywords = st.sidebar.checkbox("Enable Keyword Generation", value=True)
generate_title = st.sidebar.checkbox("Enable Title Generation", value=True)
summary_length = st.sidebar.slider("Summary Length", min_value=10, max_value=150, value=50, step=10)


st.subheader("Input Section")
uploaded_file = st.file_uploader("Upload a text file (PDF, DOCX, TXT):", type=['pdf', 'docx', 'txt'])
input_text = st.text_area("Or enter your text here:", "", height=200, max_chars=2000)

uploaded_text = ""
if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        uploaded_text = read_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        uploaded_text = read_word(uploaded_file)
    elif file_type == "text/plain":
        uploaded_text = StringIO(uploaded_file.read().decode("utf-8")).read()

input_text = input_text or uploaded_text

if st.button("Generate Output"):
    if not input_text.strip():
        st.error("Please provide input text or upload a file.")
    else:
        
        with st.spinner("🔍 Generating summary..."):
            try:
                summary = summarizer(input_text[:512], max_length=summary_length, min_length=20, truncation=True)
                summary_text = summary[0]['summary_text']
                st.text_area("Summary:", summary_text, height=200)
            except Exception as e:
                st.error(f"Summarization Error: {str(e)}")

        
        if generate_keywords:
            with st.spinner("🔑 Extracting keywords..."):
                try:
                    keywords = keyword_model.extract_keywords(input_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                    keyword_list = [kw[0] for kw in keywords]
                    st.text_area("Keywords:", ", ".join(keyword_list), height=68)
                except Exception as e:
                    st.error(f"Keyword Generation Error: {str(e)}")

        
        if generate_title:
            with st.spinner("📝 Generating title..."):
                try:
                    title_prompt = f"generate a title for the following text: {input_text[:512]}"
                    title_output = title_generator(title_prompt, max_length=10, num_return_sequences=1)
                    generated_title = title_output[0]['generated_text'].strip()
                    st.text_area("Generated Title:", generated_title, height=68)
                except Exception as e:
                    st.error(f"Title Generation Error: {str(e)}")

        st.success("🎉 Output generated successfully!")




# & "C:\Users\14har\AppData\Roaming\Python\Python312\Scripts\streamlit.exe" run index.py
