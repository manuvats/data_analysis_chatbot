import streamlit as st
from streamlit_chat import message


MAX_LENGTH_MODEL_DICT = {
    "gpt-4": 8191,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384
}


def get_text():
    """Input text by the user"""
    input_text = st.text_input(
        label="Ask me your question.",
        value="",
        key="input"
    )
    return input_text


def sidebar():
    """App sidebar content"""

    model = st.selectbox(
        label="Available Models",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
        help="""The available models. Same prompt might return different results for
        different models. Epxerimentation is recommended."""
    )

    temperature = st.slider(
        label="Temperature",
        value=0.0,
        min_value=0.,
        max_value=2.,
        step=0.01,
        help=(
            """Controls randomness. What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower values
            like 0.2 will make it more focused and deterministic.
            It is recommended to alter this or `top_n` but not both"""
        )
    )
    max_tokens = st.slider(
        label="Maximum length (tokens)",
        value=256,
        min_value=0,
        max_value=MAX_LENGTH_MODEL_DICT[model],
        step=1,
        help=(
            """The maximum number of tokens to generate in the chat completion.
            The total length of input tokens and generated tokens is limited by
            the model's context length."""
        )

    )
    top_p = st.slider(
        label="Top P",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            """An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability
            mass are considered.
            It is recommended to alter this or `temperature` but not both"""
        )
    )
    out_dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    return out_dict