import os
import validators
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Load .env file
# --------------------------------------------------

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

groq_api_key = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜",
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    groq_api_key_input = st.text_input(
        "Groq API Key",
        value=groq_api_key if groq_api_key else "",
        type="password",
    )

    generic_url = st.text_input(
        "URL",
        label_visibility="collapsed",
    )


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key_input,
    temperature=0,
)


# --------------------------------------------------
# Prompt Templates
# --------------------------------------------------

map_prompt_template = """
Write a concise summary of the following content.

Focus only on the important information and key points.

Content:
{text}

Summary:
"""

map_prompt = PromptTemplate(
    template=map_prompt_template,
    input_variables=["text"],
)


combine_prompt_template = """
You are given several summaries of different parts of a document.

Combine them into one clear and concise final summary.

The final summary should be approximately 300 words.

Do not mention that the content was divided into chunks.

Summaries:
{text}

Final Summary:
"""

combine_prompt = PromptTemplate(
    template=combine_prompt_template,
    input_variables=["text"],
)


# --------------------------------------------------
# Summarize Button
# --------------------------------------------------

if st.button("Summarize the Content from YT or Website"):

    # --------------------------------------------------
    # Validate Inputs
    # --------------------------------------------------

    if not groq_api_key_input.strip():

        st.error("Please provide your Groq API Key.")

    elif not generic_url.strip():

        st.error("Please provide a URL.")

    elif not validators.url(generic_url):

        st.error(
            "Please enter a valid URL. "
            "It can be a YouTube video URL or a website URL."
        )

    else:

        try:

            with st.spinner("Loading content..."):

                # --------------------------------------------------
                # Load YouTube or Website Data
                # --------------------------------------------------

                if (
                    "youtube.com" in generic_url
                    or "youtu.be" in generic_url
                ):

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False,
                        language=["en"],
                    )

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 "
                                "(KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        },
                    )

                docs = loader.load()


                # --------------------------------------------------
                # Check Loaded Content
                # --------------------------------------------------

                if not docs:

                    st.error(
                        "No content could be extracted from this URL."
                    )
                    st.stop()


                # --------------------------------------------------
                # Split Large Documents
                # --------------------------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=6000,
                    chunk_overlap=200,
                )

                split_docs = text_splitter.split_documents(docs)


                st.info(
                    f"Content loaded successfully. "
                    f"Created {len(split_docs)} chunks for summarization."
                )


                # --------------------------------------------------
                # Summarization Chain
                # --------------------------------------------------

                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                )


                # --------------------------------------------------
                # Generate Summary
                # --------------------------------------------------

                with st.spinner(
                    "Generating summary... This may take some time "
                    "for long videos."
                ):

                    output_summary = chain.run(split_docs)


                # --------------------------------------------------
                # Display Summary
                # --------------------------------------------------

                st.success("Summary generated successfully!")

                st.subheader("Summary")

                st.write(output_summary)


        except Exception as e:

            st.error("An error occurred while processing the content.")

            st.exception(e)