import os

os.environ["PACKAGE_ENV"] = "local"

import composo as cp
import copy
from litellm import completion
import litellm

litellm.set_verbose = True

model_mapping = {
    "OpenAI GPT-3.5 Turbo": "gpt-3.5-turbo",
    "OpenAI GPT-4 Turbo": "gpt-4-1106-preview",
    "Google Gemini Pro": "gemini-pro",
    "Cohere Command": "command",
    "Cohere Command Light": "command-light",
    "Claude V2.1": "anthropic.claude-v2:1",
    "Meta Llama 2 13b": "meta.llama2-13b-chat-v1",
    "Meta Llama 2 70b": "meta.llama2-70b-chat-v1",
}


@cp.Composo.link()
def simple_llm_call(
    model: cp.MultiChoiceStrParam(
        choices=list(model_mapping.keys()),
    ),
    temperature: cp.FloatParam(min=0.0, max=2.0),
    system_message: cp.StrParam(description="System Message."),
    conversation_history: cp.ConversationHistoryParam,
):
    messages = copy.deepcopy(conversation_history)
    messages.insert(0, {"role": "system", "content": system_message})

    prediction = (
        completion(
            model_mapping[model], messages, temperature=temperature, max_tokens=512
        )
        .choices[0]
        .message["content"]
    )

    return prediction


output = simple_llm_call(
    model="OpenAI GPT-3.5 Turbo",
    temperature=0.7,
    system_message="Be kind and helpful",
    conversation_history=[
        {
            "role": "user",
            "content": "Hi! What's good?",
        }
    ],
)
