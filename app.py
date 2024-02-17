from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
import streamlit as st

load_dotenv(find_dotenv())


def convert_datetime_to_str():
    return datetime.now().strftime("%m%d%Y_%H%M%S")


def img_2_text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text


def openai_gen_story(scenario):
    template = """
    you can generate a short story about {message}, the story should be more than 40 words, and less than 60 words
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4")
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    story_completion = chain.invoke({"message": scenario})
    return story_completion


def openai_text_2_speech(message):
    client = OpenAI()
    speech_file_path = "speech_"+convert_datetime_to_str()+".mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=message
    )
    response.stream_to_file(speech_file_path)  
    return speech_file_path


def main():
    st.set_page_config(page_title="image to audio demo")
    st.header("Generates Speech From Images with Hugging Face, LangChain, OpenAI, and Streamlit")
    uploaded_file = st.file_uploader("Upload an image...")
    
    if uploaded_file:
        bytes_val = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as _f:
            _f.write(bytes_val)
        st.image(uploaded_file, caption='Image Uploaded.')
        
        scenario = img_2_text(uploaded_file.name)
        stroy = openai_gen_story(scenario)
        audio_path = openai_text_2_speech(stroy)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("stroy"):
            st.write(stroy)
            
        st.audio(audio_path)
    else:
        st.write("Could not find the iamge, pls try again")
        
        
if __name__ == '__main__':
    main()    