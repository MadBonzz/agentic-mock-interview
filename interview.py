from llama_cpp import Llama
import pymupdf4llm
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import time
import os
import prompts

pdf_path = 'uploaded_file.pdf'
embedding_model_path = 'C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf'
llm_path = 'C:/Users/shour/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
collection_name = 'resume'
curr_dir = os.getcwd()

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

llm = Llama(
  model_path=llm_path,
  n_ctx=16384,
  n_gpu_layers=-1,
  offload_kqv=True,
  n_threads=16
)

first_response = llm.create_chat_completion(
     messages=[
        {
            "role": "system",
            "content": "You are an AI interviewer following a structured process."
        },
        {
            "role": "user",
            "content": prompts.initial_prompt_template.format(
                context="\n\n".join([row.payload['text'] for row in search_result]),
                job_description=job_description
            )
        }
    ],
    tools=[
       {
            "type": "function",
            "function": {
                "name": "Information-Parser",
                "parameters": {
                    "type": "object",
                    "title": "Information-Parser",
                    "properties": {
                            "question": {
                            "title": "Question",
                            "type": "string",
                            "description": "The interview question being asked to the candidate."
                        },
                        "question_category": {
                            "title": "Question Category",
                            "type": "string",
                            "description": "Category of the question, must be one of: Personality, Technical Concept, Experience.",
                            "enum": ["Personality", "Technical Concept", "Experience"]
                        }
                    },
                    "required": [
                        "question",
                        "question_category"
                    ]
                }
            }
        }  
    ],
    tool_choice={"type": "function", "function": {"name": "Information-Parser"}},
    temperature=1,
)

initial_question = json.loads(first_response["choices"][0]['message']["function_call"]["arguments"])
print(initial_question)
question = initial_question['question']

print(question)
answer = ""
answer = input("Enter your answer : ")

while answer != "end":
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an AI interviewer following a structured process."
            },
            {
                "role": "user",
                "content": prompts.prompt_template.format(
                    context="\n\n".join([row.payload['text'] for row in search_result]),
                    answer=f"Question : {question} Answer : {answer}"
                )
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "InterviewEvaluation",
                    "parameters": {
                        "type": "object",
                        "title": "InterviewEvaluation",
                        "properties": {
                            "technical_ability": {
                                "title": "Technical Ability",
                                "type": "integer",
                                "description": "Rating for technical correctness and depth of answer (0-10).",
                                "minimum": 0,
                                "maximum": 10
                            },
                            "language_clarity": {
                                "title": "Language & Clarity",
                                "type": "integer",
                                "description": "Rating for communication, structure, and articulation (0-10).",
                                "minimum": 0,
                                "maximum": 10
                            },
                            "depth_of_knowledge": {
                                "title": "Depth of Knowledge",
                                "type": "integer",
                                "description": "Rating for ability to explain concepts beyond surface level (0-10).",
                                "minimum": 0,
                                "maximum": 10
                            },
                            "review": {
                                "title": "Review",
                                "type": "string",
                                "description": "Detailed feedback including strengths and areas for improvement."
                            },
                            "satisfaction_level": {
                                "title": "Satisfaction Level",
                                "type": "string",
                                "description": "Overall assessment of the candidate’s answer.",
                                "enum": ["Highly Satisfactory", "Satisfactory", "Needs Improvement", "Unsatisfactory"]
                            },
                            "next_step": {
                                "title": "Next Step",
                                "type": "string",
                                "description": "Determines the next action to take based on the answer.",
                                "enum": ["Ask a deeper question", "Move to a new topic", "Prompt for a better answer"]
                            },
                            "question": {
                                "title": "Question",
                                "type": "string",
                                "description": "The interview question being asked to the candidate."
                            },
                            "question_category": {
                                "title": "Question Category",
                                "type": "string",
                                "description": "Category of the question, must be one of: Personality, Technical Concept, Experience.",
                                "enum": ["Personality", "Technical Concept", "Experience"]
                            }
                        },
                        "required": [
                            "technical_ability",
                            "language_clarity",
                            "depth_of_knowledge",
                            "review",
                            "satisfaction_level",
                            "next_step",
                            "question",
                            "question_category"
                        ]
                    }
                }
            }
        ],
        tool_choice={"type": "function", "function": {"name": "InterviewEvaluation"}},
        temperature=1,
    )

    parsed_response = json.loads(response["choices"][0]['message']["function_call"]["arguments"])
    print(parsed_response)
    question = parsed_response['question']
    print(question)
    answer = input("Enter your answer : ")