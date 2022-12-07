from flask import Flask, jsonify, request, render_template, redirect
import openai as ai
import json
import os

print("** Loading API Key")
ai.api_key = os.getenv("OPENAI_API_KEY")
model_used = "text-davinci-002"

app = Flask(__name__)


@app.route('/', methods=['POST'])
def hello_world():
    # company_name = input("Company Name: "/)
    body = request.get_json()
    print(body)
    project_title = body["project_title"]
    contact_person = body["contact_person"]
    your_name = body["your_name"]
    price_per_hour = str(body["price_per_hour"])

    prompt = ("Write a cover letter to "+contact_person+" from "+your_name+" for the project " +
              project_title + ". My best quotation would me "+price_per_hour + " per hour.")

    # print(prompt)
    response = ai.Completion.create(
        engine=model_used,
        # engine="text-davinci-002", # OpenAI has made four text completion engines available, named davinci, ada, babbage and curie. We are using davinci, which is the most capable of the four.
        prompt=prompt,  # The text file we use as input (step 3)
        # how many maximum characters the text will consists of.
        max_tokens=int(1949),
        temperature=0.99,
        # temperature=int(temperature), # a number between 0 and 1 that determines how many creative risks the engine takes when generating text.,
        # an alternative way to control the originality and creativity of the generated text.
        top_p=int(1),
        n=1,  # number of predictions to generate
        # a number between 0 and 1. The higher this value the model will make a bigger effort in not repeating itself.
        frequency_penalty=0.9,
        # a number between 0 and 1. The higher this value the model will make a bigger effort in talking about new topics.
        presence_penalty=0.9
    )

    text = response['choices'][0]['text']

    return {"Prompt": prompt, "Response": text}
