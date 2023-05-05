from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message

from kbqa_test import ins
from yuyinjishu import yuyin
st.set_page_config(

    page_title="智能AI机器人 华陀-张仲景",
    page_icon=":robot:"
)


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
    model = model.eval()
    return tokenizer, model


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

weans = ins(prompt_text)

def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")



        with st.empty():
            for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
                query, response = history[-1]
                answer = "请先听听华陀的意见，谢谢！"
                if input == prompt_text:
                    st.write(f"<p>张仲景：{answer}</p><p>华陀：{response}</p>", unsafe_allow_html=True)

                else:
                    st.write(f"<p>张仲景：{input}</p><p>华陀：{response}</p>", unsafe_allow_html=True)


        if input == prompt_text:
            all = '张仲景说'+ answer + '华陀说' + response
            yuyin(all)
        else:
            all1 = '张仲景说'+ input + '华陀说' + response
            yuyin(all1)
        audio_file = open('audio.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')




    return history


container = st.container()

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 2048, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation

        st.session_state["state"] = predict(weans, max_length, top_p, temperature, st.session_state["state"])


#cd Pycharm/QASystemOnMedicalGraphpro      streamlit run web_demo2copy.py
