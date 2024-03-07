import composo as cp
import litellm

llm_1_prompt = """\
You are a medical expert. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. 

Here are a few examples:

Original Question: A 35-year-old male presents to his primary care physician with complaints of seasonal allergies. He has been using intranasal vasoconstrictors several times per day for several weeks. What is a likely sequela of the chronic use of topical nasal decongestants?
Stepback Question: What are the principles behind the use of topical nasal decongestants & all the potential side effects of chronic use?

Original Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?
Stepback Question: What are the potential causes of these symptoms & signs?\
"""

stepback_generation_user_message = "Original Question: {user_message}"

llm_2_prompt = """\
You are a medical expert. Your task is to answer the user's question accurately and concisely.\
    
It may be useful to know that the question is related to the wider question of: {user_message}. However, your task is not to answer this wider question, focus on the user's question.\
"""

llm_3_prompt = """\
You are a medical expert. Your task is to answer the user's question accurately and concisely. You'll be provided with a question and answer pair related to the topic that may be useful in answering the user's question.

Question:{stepback_question}
Answer: {stepback_answer}\
"""


@cp.Composo.link()
def medical_advice_bot(
    user_message: cp.StrParam(description="This is the user's query to the AI doctor"),
    llm_1_prompt: cp.StrParam(
        description="This is the system message for LLM 1, which uses the user's query to generate a 'stepback question'. This is an abstraction of the original query which helps the LLM to think through its answer and can lead to more accurate responses. For example, if the question were a specific question related to physics, a good stepback question might be 'What are the physics principles important in this question?'."
    ) = llm_1_prompt,
    llm_2_prompt: cp.StrParam(
        description="This is the system message for LLM 2, which takes as inputs the user's original query & the generated stepback question, and then outputs the answer to this stepback question."
    ) = llm_2_prompt,
    llm_3_prompt: cp.StrParam(
        description="This is the system message for LLM 3, which takes as inputs the user's original query, the stepback question, the stepback answer, and then outputs the final answer back to the user."
    ) = llm_3_prompt,
    temperature: cp.FloatParam(
        description="This can be thought of as the level of creativity, randomness or determinism displayed. Temperature is a value between 0 and 2. Higher values like 1.6 will make the output more random, while lower values like 0.2 will make the output more focussed and deterministic.",
        min=0,
        max=2,
    ) = 0.7,
):
    """
    A demo application which uses step-back prompting to answer medical questions.
    \n\n
    \n\n
    For further explanation please refer to the Quickstart section in our documentation [here](https://www.notion.so/Composo-documentation-c8188c36fd7a4f7d9d45b2faa436316f?pvs=4).
    """

    # Generate the stepback question
    stepback_question = (
        litellm.completion(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": llm_1_prompt},
                {
                    "role": "user",
                    "content": stepback_generation_user_message.format(
                        user_message=user_message
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=512,
        )
        .choices[0]
        .message["content"]
    )

    if ":" in stepback_question:
        stepback_question = stepback_question.split(":")[1].strip()

    print("Generated stepback question = ", stepback_question)

    # Answer the stepback question
    stepback_answer = (
        litellm.completion(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": llm_2_prompt.format(
                        user_message=user_message,
                    ),
                },
                {
                    "role": "user",
                    "content": stepback_question,
                },
            ],
            temperature=temperature,
            max_tokens=512,
        )
        .choices[0]
        .message["content"]
    )

    print("Generated stepback answer = ", stepback_answer)

    # Generate the final response
    final_answer = (
        litellm.completion(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": llm_3_prompt.format(
                        stepback_question=stepback_question,
                        stepback_answer=stepback_answer,
                    ),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=temperature,
            max_tokens=512,
        )
        .choices[0]
        .message["content"]
    )

    return final_answer


answer = medical_advice_bot(
    ""
)

print("Final answer = ", answer)
