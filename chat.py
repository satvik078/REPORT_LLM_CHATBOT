import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./mac_drone_model_merged"

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ⭐ NEW — Casual Intent Handler
def casual_response(user_input):

    text = user_input.lower().strip()

    greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
    jokes = ["tell me a joke", "joke", "make me laugh"]

    if text in greetings:
        return "Hello 🙂 How can I help you today?"

    if text in jokes:
        return "Why did the surveillance drone get promoted? Because it always rose above the rest 🚁😄"

    if "how are you" in text:
        return "I'm functioning perfectly and ready to assist you with your drone project."

    return None


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16
    )

    model.to(device)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def generate_response(user_input):
    system_prompt = f"""
                     You are an expert assistant for a Wildlife Surveillance Drone System.

                     Answer ONLY about the concept asked.
                     Do not switch topic.
 
              """

    prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
        **inputs,
        max_new_tokens=180,
        min_new_tokens=60,
        do_sample=True,
        temperature=0.45,
        top_p=0.8,
        repetition_penalty=1.2,
        length_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()

    return response


st.title("🛰️ Wildlife Drone Surveillance AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask about your drone project...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})

    with st.spinner("Thinking..."):

        # ⭐ NEW — Check Casual Intent First
        casual = casual_response(user_input)

        if casual is not None:
            reply = casual
        else:
            reply = generate_response(user_input)

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role":"assistant","content":reply})