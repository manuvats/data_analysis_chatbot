import streamlit as st
from streamlit_chat import message
import pandas as pd
from utils import get_text, sidebar
from llm_utils import chat_with_data_api


def chat_with_data():

    st.title("Chat with, query and plot your own data")

    with st.sidebar:
        model_params = sidebar()
        memory_window = st.slider(
            label="Memory Window",
            value=3,
            min_value=1,
            max_value=10,
            step=1,
            help=(
                """The size of history chats that is kept for context. A value of, say,
                3, keeps the last three pairs of promtps and reponses, i.e. the last
                6 messages in the history."""
            )
        )

    api_key = st.text_input("Enter your OpenAI API key here:", type = "password")

    uploaded_file = st.file_uploader(label="Choose file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        prompt = f"""You are a python expert. You will be given questions for
            manipulating an input dataframe.
            The available columns are: `{df.columns}`.
            Use them for extracting the relevant data.
        """
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "system", "content": prompt}]
    else:
        df = pd.DataFrame([])

    # Storing the chat
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Please upload your data"]

    if "past" not in st.session_state:
        st.session_state["past"] = []

    user_input = get_text()

    if ((len(st.session_state["past"]) > 0)
            and (user_input == st.session_state["past"][-1])):
        user_input = ""

    if ("messages" in st.session_state) and \
            (len(st.session_state["messages"]) > 2 * memory_window):
        # Keep only the system prompt and the last `memory_window` prompts/answers
        st.session_state["messages"] = (
            # the first one is always the system prompt
            [st.session_state["messages"][0]]
            + st.session_state["messages"][-(2 * memory_window - 2):]
        )
    warning_msg = []
    if user_input:
        if api_key=="":
            warning_msg.append("Please enter your OpenAI API key ⚠️ \n")
        if df.empty:
            warning_msg.append("Dataframe is empty, upload a valid file ⚠️ \n")
        if not warning_msg:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            response = chat_with_data_api(df, api_key, **model_params)
            st.session_state.past.append(user_input)
            if response is not None:
                st.session_state.generated.append(response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response})
    
    if warning_msg:
        for msg in warning_msg:
            st.warning(msg + "\n")
    
    warning_msg.clear()

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            if i - 1 >= 0:
                message(
                    st.session_state["past"][i - 1],
                    is_user=True,
                    key=str(i) + "_user"
                )
    


if __name__ == "__main__":
    chat_with_data()
