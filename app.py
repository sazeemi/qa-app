import streamlit as st
from transformers import pipeline

st.title("Question Answering App — Saman Azeemi | B04-0923-000053")

@st.cache_resource
def load_model():
    return pipeline("question-answering",
                    model="distilbert-base-cased-distilled-squad")

qa = load_model()

context = st.text_area("Context passage", height=200)
question = st.text_input("Your question")

if st.button("Get answer"):
    if not context.strip() or not question.strip():
        st.warning("Please enter both a context and a question.")
    else:
        with st.spinner("Thinking..."):
            result = qa(question=question, context=context)
        st.success(f"Answer: {result['answer']}")
        st.write(f"Confidence score: {result['score']:.4f} ({result['score']*100:.1f}%)")
        st.progress(min(max(float(result['score']), 0.0), 1.0))
