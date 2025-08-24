#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 15:28:25 2025

@author: wintonpierson
"""

import os
import streamlit as st

st.set_page_config(page_title="Winton's ChatBot", page_icon="💬")
st.title("💬 Chatbot")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    if st.button("Clear chat history"):
        st.session_state.messages = []

# --- Session State for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything."}]

# --- Show history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Core LLM streaming helper ---
def stream_reply(user_input: str):
    """
    Yields chunks of the assistant reply.
    If OPENAI_API_KEY is set and the openai library is installed, it streams from OpenAI.
    Otherwise, it falls back to a simple local echo.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if api_key:
        try:
            # OpenAI SDK v1.x style
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            stream = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *st.session_state.messages,
                    {"role": "user", "content": user_input},
                ],
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
            return
        except Exception as e:
            # If anything goes wrong, fall back gracefully
            yield f"_LLM error, falling back locally:_ {e}\n\n"

    # Fallback: simple local response (no API key needed)
    intro = "I don’t have an API key configured, so here’s a local demo reply.\n\n"
    yield intro + "You said: "
    for ch in user_input:
        yield ch

# --- Chat input + response ---
if user_input := st.chat_input("Type your message"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant message
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        for chunk in stream_reply(user_input):
            full += chunk
            placeholder.markdown(full)
    st.session_state.messages.append({"role": "assistant", "content": full})
