import openai
import composo as cp
import copy

@cp.Composo.link()
def simple_llm_call(
    model: cp.MultiChoiceStrParam(choices=["gpt-3.5-turbo", "gpt-4"]),
    temperature: cp.FloatParam(min=0.0, max=2.0),
    system_message: cp.StrParam(
        description="System Message. Use {text}, {length} and {audience} to insert variables."
    ),
    conversation_history: cp.ConversationHistoryParam,
):
    messages = copy.deepcopy(conversation_history)
    messages.insert(0, {"role": "system", "content": system_message})

    prediction = (
        openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=256,
        )
        .choices[0]
        .message["content"]
    )
    return prediction


output = simple_llm_call(
    model="gpt-3.5-turbo",
    temperature=0.7,
    system_message="Summarise the provided text",
    conversation_history=[
        {
            "role": "user",
            "content": "The following is a summary of the provided text: ",
        }
    ],
)
