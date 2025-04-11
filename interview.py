from llama_cpp import Llama
import pymupdf4llm
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os
from openai import OpenAI
from generate_response import extraction, initial_question, general_question
from report import get_df, get_review, overall_review, generate_pdf

pdf_path = 'resumes/uploaded_file.pdf'
embedding_model_path = 'C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf'
llm_id = 'meta-llama-3.1-8b-instruct'
collection_name = 'resume'
server_url = 'http://127.0.0.1:1234/v1'
review_path = "output.txt"
report_path = "output.pdf"
curr_dir = os.getcwd()

gen_client = OpenAI(base_url=server_url, api_key="lm-studio")

md_text = pymupdf4llm.to_markdown(pdf_path)

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[('##', 'header1'), ('####', 'header2')]
)

texts = splitter.split_text(md_text)

embed_model = Llama(model_path=embedding_model_path, 
                    embedding=True,
                    verbose=False)

chunks = []
for i in range(len(texts)):
    chunk = dict()
    chunk['content']   = texts[i].page_content
    chunk['length']    = len(texts[i].page_content)
    chunk['embedding'] = embed_model.create_embedding(texts[i].page_content)['data'][0]['embedding']
    chunks.append(chunk)

file_name = 'chunks.json'
with open(file_name, 'w') as file:
    json.dump(chunks, file, indent=4)

client = QdrantClient(path="embeddings")

try:
    if client.get_collection(collection_name):
            client.delete_collection(collection_name=collection_name)
except ValueError:
    print("Collection not found")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

embeddings = [chunk['embedding'] for chunk in chunks]

points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['embedding'],
    payload = {
      "text": chunk['content']
    }
  )
  for chunk in chunks]

operation_info = client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points
)

job_description = """
Job Description:
We are seeking a skilled and experienced Angular/.NET/Ionic Developer to join our development team. In this role, you will be responsible for developing and maintaining web applications using Angular for the front-end, .NET for the back-end, and Ionic for mobile application development. You will collaborate with cross-functional teams to design, develop, and deploy high-quality software solutions.

Share your resume to inshita@intuitiveapps.com with these details-
• Experience in .NET-
• Experience in Angular-
• Experience in Ionic-
• Current CTC
• Expected CTC
• Notice period (Official/Negotiable up to)
• Last working Day
• Amount of offer (If holding any)
• Current Location
• Preferred Location

Responsibilities-
• Develop and maintain web applications using Angular, .NET, and Ionic frameworks.(Flutter is good to have)
• Collaborate with designers, back-end developers, and stakeholders to gather requirements and translate them into technical specifications.
• Write clean, scalable, and maintainable code.
• Implement responsive designs and ensure cross-browser compatibility.
• Integrate RESTful APIs and third-party services into the applications.
• Conduct thorough testing and debugging to ensure high-quality software delivery.
• Optimize applications for maximum speed and scalability.
• Stay up-to-date with emerging technologies and industry trends to continuously improve development processes and techniques.
• Participate in code reviews and provide constructive feedback to team members.
• Assist in troubleshooting and resolving issues reported by end-users.

Requirements-
• Proven experience in developing web applications using Angular, .NET, and Ionic frameworks.
• Strong understanding of front-end technologies such as HTML5, CSS3, and JavaScript.
• Proficient in C# and .NET framework for back-end development.
• Familiarity with RESTful APIs and JSON.
• Experience with version control systems, such as Git.
• Knowledge of responsive design principles and mobile-first development.
• Understanding of software development best practices, including code optimization, debugging, and testing.
• Ability to work independently and collaboratively in a team environment.
• Excellent problem-solving and communication skills.

Skills: angular,.net,ionic framework,flutter,mobile app development
"""

query_vector = embed_model.create_embedding(job_description)['data'][0]['embedding']
search_result = client.search(
  collection_name=collection_name,
  query_vector=query_vector,
  limit=3
)

concepts = extraction(gen_client, llm_id, search_result, job_description)
concepts = eval(concepts)["concepts"]
if isinstance(concepts, str):
    concepts = eval(concepts)
print(concepts)

first_response = initial_question(gen_client, llm_id, search_result, concepts)

question = first_response['question']
category = first_response["question_category"]
concept_asked = first_response["concept_asked"]
print(category)
print(concept_asked)
print(question)

file = open(review_path, "w")
file.write(str(first_response) + "\n")

if concept_asked in concepts:
    concepts.remove(concept_asked)

answer = ""
answer = input("Enter your answer : ")

questions = []
questions.append(question)

while answer != "end":
    parameters_dict = general_question(gen_client, llm_id, questions, concepts, concept_asked, question, answer)
    question = parameters_dict['next_question']
    while question in questions:
        parameters_dict = general_question(gen_client, llm_id, questions, concepts, concept_asked, question, answer)
        question = parameters_dict['next_question']
    print(parameters_dict)
    concept_asked = parameters_dict["new_concept"]
    questions.append(question)
    file.write(str(parameters_dict) + "\n")
    if concept_asked in concepts:
        concepts.remove(concept_asked)
    print(question)
    answer = input("Enter your answer : ")

file.close()

df = get_df(review_path)
eval_string = get_review(df)
final_review = overall_review(gen_client, llm_id, eval_string)
generate_pdf(df, report_path, final_review)