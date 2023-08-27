import os
import numpy as np
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import speech_recognition as sr
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import json

PAT = '5881bd40da7a40c08e8d3c0faa1c4aee'
USER_ID = 'dj7fu4s80fz1'
XI_API_KEY = "adece68de32121faed9c205107d415d1"
APP_ID = 'Sentiment-analysis-1'
WORKFLOW_ID = 'workflow-c2d009'

def message_dict(role, content):
    return {"role": role, "content": content}

def clarifAI_request(content): 
    TEXT = """You are customer care at airkom which is a indian telecommunication company, an automated service to\
    provide recharge plans based on the customer's query. \
    You first greet the customer, then interact with them about their problem.\
    Ask for which course they would like to gain information about.\
    Once the course has been selected by the recharge plans \
    then finally you ask them for the mode of payment.\
    You should respond in a short, very conversational friendly style. \
    airkom’s prepaid plans come with benefits such as doorstep KYC, paperless process, and quick activation. Customers can also enjoy free data coupons on select recharges via the Airkom Thanks app \
    Airkom’s new prepaid plan prices now begin at Rs 99 instead of the Rs 79 plan. Meanwhile, unlimited voice bundles now start at Rs 179, instead of the Rs 149 plan, going all the way up to Rs 2,999 instead of Rs 2,498 for the annual plan. Meanwhile, data top-ups are now priced at Rs 58 (3GB), Rs 118 (12GB) and Rs 301 (50GB). \
    Rs 58 – The recharge plan comes with 28 days of validity and offers 3GB of data. Rs 118 – The Rs 98 4G data recharge plan comes with 12GB of data. The pack is valid till the validity of the primary recharge plan. Rs 148 – The 4G data plan offers 15GB of data and has the same validity as your existing pack. Rs 301 – The Rs 301 prepaid 4G recharge plan offers 50GB of data, which is valid until the validity of the primary pack is over. \
    \
    
     \
    Airkom’s broadband service provides high-speed internet access with unlimited data and speeds up to 1 Gbps. Airkom’s DTH service offers a wide selection of channels and packages to suit the needs of different customers.
    \
    query from customer : {recognized_text}"""

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_workflow_results_response = stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=userDataObject,
            workflow_id=WORKFLOW_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=TEXT)
                    )
                )]
        ),
        metadata=metadata
    )
    if post_workflow_results_response.status.code != status_code_pb2.SUCCESS:
        print(post_workflow_results_response.status)
        raise Exception("Post workflow results failed, status: " +
                        post_workflow_results_response.status.description)

    # We'll get one WorkflowResult for each input we used above. Because of one input, we have here one WorkflowResult
    results = post_workflow_results_response.results
    results = results[0]
    results = results.outputs.pop()
    results = results.data.text.raw
    # Each model we have in the workflow will produce one output.
    return results

def audio_recorder():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "audio-recorder-src")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)
    raw_audio_data = st_audiorec() 
    if raw_audio_data:
        wav_bytes = None     
        if isinstance(raw_audio_data, dict): 
            ind, raw_audio_data = zip(*raw_audio_data['arr'].items())
            ind = np.array(ind, dtype=int)
            raw_audio_data = np.array(raw_audio_data)
            sorted_ints = raw_audio_data[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()
        return wav_bytes
    return None

def chat():

    if "messages" not in st.session_state:
        st.session_state.messages = list()
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    if audio_recorder_wav_fmt := audio_recorder():

        if audio_recorder_wav_fmt!=None and len(audio_recorder_wav_fmt) > 0:
            audio_file = open("audio.wav",'wb')
            audio_file.write(audio_recorder_wav_fmt)
            audio_file.close()
            r = sr.Recognizer()                                                      
            audio = sr.AudioFile("audio.wav")
            try:
                if "messages" not in st.session_state:
                    st.session_state["messages"] = list()
                with audio as source:
                    audio = r.record(audio)                  
                    content = r.recognize_google(audio)
                    st.session_state["messages"].append(message_dict("user",content))
                    print(st.session_state.messages)
                    with st.chat_message("user"):
                        st.markdown(content)
                    response = clarifAI_request(content)
                    st.session_state["messages"].append(message_dict( "assistant", response))
                    # st.experimental_rerun()
            except sr.UnknownValueError:
                st.session_state["messages"].append(message_dict("assistant","Sorry, Can you please rephrase that?"))
            except sr.RequestError as e:
                st.session_state["messages"].append(message_dict("assistant",f"Could not retrieve results. {e}"))
            os.remove("audio.wav")
    


def home():
    global chat_convo
    st.set_page_config(
        page_title="CareLinkAI - Web"
    )

    style = st.markdown("""
    <style>
        div.stButton > button:first-child {
        background-color: #0000FF;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #FF0000;
        color:##ff99ff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("CareLink - Web")
    st.write("\tSmart AI-Driven Solutions for Telecom Operation and Customer Support")

    st.header("Guidelines")
    guidelines = """
    \t- Be respectful and courteous to our support agents.
    \t- Provide clear and concise information about your issue.
    \t- Avoid sharing sensitive information in public chats.
    """
    st.write(guidelines)

    st.header("Join in conversation with us!")
    _, _, centered, _, _ = st.columns(5)
    chat_button = st.button("Start your conversation with our CareLink Agent!")
    
    if chat_button:
        st.session_state.active_page = "Chat"
        st.write(st.session_state.active_page)
        st.experimental_rerun()
        

    st.header("FAQ")
    faq = {
        "How do I reset my password?": "You can reset your password by visiting the 'Forgot Password' link...",
        "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, and PayPal...",
        # Add more FAQ items here
    }

    for question, answer in faq.items():
        with st.expander(question):
            st.write(answer)

if st.session_state.get('active_page')==None or st.session_state.active_page == 'Home':
    home()
if st.session_state.get('active_page')!=None and st.session_state.active_page == 'Chat':
    chat()
