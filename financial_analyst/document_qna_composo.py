from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import tiktoken
import composo as cp
import time


db = FAISS.load_local("embeddings/2022-alphabet-annual-report", OpenAIEmbeddings())

default_system_message = "You are a helpful customer service assistant. Use the snippets provided to answer the user's question: {retrieved_snippets}"


@cp.Composo.link()
def document_qa(
    temperature: cp.FloatParam(
        description="This can be thought of as the level of creativity, randomness or determinism displayed. Temperature is a value between 0 and 2. Higher values like 1.6 will make the output more random, while lower values like 0.2 will make the output more focussed and deterministic.",
        min=0,
        max=2,
    ),
    system_message: cp.StrParam(
        description="This is the instructions that 'program' the model for how it should behave in response to user inputs"
    ),
    number_of_retrieved_snippets: cp.IntParam(
        description="This is the number of text snippets (search results) from the pdf that are passed to the model to summarise back to user",
        min=1,
        max=5,
    ),
    user_message: cp.StrParam(description="This is the user's input"),
):
    """
    A demo application which answer's a user's queries about Alphabet's 2022 annual report using retrieval augmented generation (RAG).
    \n\n
    \n\n
    For further explanation please refer to the Quickstart section in our documentation [here](https://www.notion.so/Composo-documentation-c8188c36fd7a4f7d9d45b2faa436316f?pvs=4).
    """
    if len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(user_message)) > 256:
        return "This demo app has a 256 token limit for queries"

    if len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(system_message)) > 256:
        return "This demo app has a 256 token limit for system messages"

    docs = db.similarity_search(user_message, k=number_of_retrieved_snippets)

    retrieved_snippets = (
        "\n```snippet\n"
        + "\n```\n```snippet\n".join(x.page_content for x in docs)
        + "\n```"
    )

    messages = [
        {
            "role": "system",
            "content": system_message.format(retrieved_snippets=retrieved_snippets),
        },
        {"role": "user", "content": user_message},
    ]

    time.sleep(5)

    return (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=256,
        )
        .choices[0]
        .message["content"]
    )


answer = document_qa(
    1,
    default_system_message,
    3,
    "How is Google helping people find new ways to empower people to stay on top of their wellness",
)
print(answer)
