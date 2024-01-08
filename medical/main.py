import composo as cp
import openai

stepback_generation_system_message = f"""\
You are a medical expert. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. 

Here are a few examples:

Original Question: A 35-year-old male presents to his primary care physician with complaints of seasonal allergies. He has been using intranasal vasoconstrictors several times per day for several weeks. What is a likely sequela of the chronic use of topical nasal decongestants?
Stepback Question: What are the principles behind the use of topical nasal decongestants & all the potential side effects of chronic use?

Original Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?
Stepback Question: What are the potential causes of these symptoms & signs?\
"""

stepback_generation_user_message = """\
Original Question: {original_question}\
"""

stepback_answer_system_message = f"""\
You are a medical expert. Your task is to answer the user's question accurately and concisely.\
"""

stepback_answer_user_message = """\
{stepback_question}\
"""

final_system_message = """\
You are a medical expert. Your task is to answer the user's question accurately and concisely. You'll be provided with a question and answer pair related to the topic that may be useful in answering the user's question.

Question:{stepback_question}
Answer: {stepback_answer}\
"""

final_user_message = """\
{original_question}\
"""


@cp.Composo.link(api_key="cp-0CB7TAY350QG53D07P4065962Y95N")
def medical_advice_bot(
    original_question: cp.StrParam,
    stepback_generation_system_message: cp.StrParam = stepback_generation_system_message,
    stepback_generation_user_message: cp.StrParam = stepback_generation_user_message,
    stepback_answer_system_message: cp.StrParam = stepback_answer_system_message,
    stepback_answer_user_message: cp.StrParam = stepback_answer_user_message,
    final_system_message: cp.StrParam = final_system_message,
    final_user_message: cp.StrParam = final_user_message,
    temperature: cp.FloatParam = 0.7,
):
    # Generate the stepback question
    stepback_question = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": stepback_generation_system_message},
                {
                    "role": "user",
                    "content": stepback_generation_user_message.format(
                        original_question=original_question
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
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": stepback_answer_system_message,
                },
                {
                    "role": "user",
                    "content": stepback_answer_user_message.format(
                        stepback_question=stepback_question
                    ),
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
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": final_system_message.format(
                        stepback_question=stepback_question,
                        stepback_answer=stepback_answer,
                    ),
                },
                {
                    "role": "user",
                    "content": final_user_message.format(original_question=original_question),
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
    "A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby?"
)

print("Final answer = ", answer)
