import html
import streamlit as st
from transformers import pipeline

MODEL = "distilbert-base-cased-distilled-squad"

st.set_page_config(page_title="Ask Your Text | Saman Azeemi", page_icon="🔎", layout="centered")

st.markdown("""
<style>
h1 {
    background: linear-gradient(90deg, #7f5af0, #2cb1bc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
.tagline { opacity: .75; margin-top: -10px; margin-bottom: 18px; }
.card {
    background: rgba(127,127,127,.12);
    border-left: 6px solid var(--accent, #7f5af0);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}
.card .label { font-size: 12px; letter-spacing: 2px; opacity: .6; }
.card .answer { font-size: 28px; font-weight: 700; }
.ctx { background: rgba(127,127,127,.08); border-radius: 10px; padding: 12px 16px; line-height: 1.7; }
.ctx mark { background: #ffd54f; color: #111; padding: 1px 5px; border-radius: 5px; font-weight: 600; }
.footer { text-align: center; opacity: .5; font-size: 12px; margin-top: 30px; }
div.stButton > button[kind="primary"] { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model (first run only)...")
def load_model():
    return pipeline("question-answering", model=MODEL)


qa = load_model()

EXAMPLES = {
    "📚 Hamlet": ("Hamlet was written by William Shakespeare around 1600.",
                 "Who wrote Hamlet?"),
    "🏙️ Sindh": ("Lahore is the capital of Punjab. Karachi is the capital of Sindh.",
                "What is the capital of Sindh?"),
    "🤔 Not in text": ("Lahore is the capital of Punjab. Karachi is the capital of Sindh.",
                      "Who is the president of France?"),
}

st.session_state.setdefault("ctx", "")
st.session_state.setdefault("q", "")


def use_example(name):
    st.session_state.ctx, st.session_state.q = EXAMPLES[name]


# Name + ID must appear in st.title
st.title("Ask Your Text 🔎 — Saman Azeemi | B04-0923-000053")
st.markdown('<div class="tagline">Paste a passage, ask a question, and the model '
            'finds the answer inside it — with a confidence score.</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("Extractive question answering: the model picks the answer **from your text**, "
             "it does not make one up from memory.")
    st.caption(f"Model: `{MODEL}`")
    st.divider()
    st.subheader("Try an example")
    for name in EXAMPLES:
        st.button(name, on_click=use_example, args=(name,), use_container_width=True)

st.text_area("📄 Context passage", key="ctx", height=170, max_chars=5000,
             placeholder="Paste the text that contains the answer...")
st.text_input("❓ Your question", key="q", placeholder="e.g. Who wrote Hamlet?")

if st.button("Get answer ✨", type="primary", use_container_width=True):
    context, question = st.session_state.ctx, st.session_state.q
    if not context.strip() or not question.strip():
        st.warning("Please enter both a context and a question.")
    else:
        try:
            with st.spinner("Thinking..."):
                result = qa(question=question, context=context)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        else:
            score = float(result["score"])
            if score >= 0.7:
                color, verdict = "#2ecc71", "High confidence"
            elif score >= 0.3:
                color, verdict = "#f5a623", "Medium confidence"
            else:
                color, verdict = "#e74c3c", "Low confidence"

            st.markdown(
                f'<div class="card" style="--accent:{color}">'
                f'<div class="label">ANSWER</div>'
                f'<div class="answer">{html.escape(result["answer"])}</div></div>',
                unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Confidence score", f"{score:.1%}")
            c2.metric("Verdict", verdict)
            st.progress(min(max(score, 0.0), 1.0))

            if score < 0.3:
                st.info("The model is unsure. The answer may not be in the text, "
                        "so please verify it.")

            s, e = result["start"], result["end"]
            st.markdown("##### Where it found the answer")
            st.markdown(
                '<div class="ctx">'
                + html.escape(context[:s]) + "<mark>" + html.escape(context[s:e])
                + "</mark>" + html.escape(context[e:]) + "</div>",
                unsafe_allow_html=True)

st.markdown('<div class="footer">Built with Streamlit + Hugging Face · CS4106 Quiz</div>',
            unsafe_allow_html=True)
