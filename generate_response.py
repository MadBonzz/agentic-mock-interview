import re
import json
from openai import OpenAI
from formats import initial_schema, general_schema, initial_prompt_template, prompt_template, system_prompt, extraction_schema, data_extraction_template, duplicate_question_template

def extraction(gen_client : OpenAI, llm_id : str, search_result, job_description):
    context = gen_client.chat.completions.create(
        model=llm_id,
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": data_extraction_template.format(
                    context="\n\n".join([row.payload['text'] for row in search_result]),
                    job_description=job_description
                )
            }
        ],
        tools=[extraction_schema],
        tool_choice={"type" : "function", "function" : {"name" : "Data-Extraction"}},
        temperature=0.6,
    )
    return context.choices[0].message.tool_calls[0].function.arguments

def initial_question(gen_client : OpenAI, llm_id : str, search_result, concepts : list[str]):
    first_response = gen_client.chat.completions.create(
        model=llm_id,
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": initial_prompt_template.format(
                    context="\n\n".join([row.payload['text'] for row in search_result]),
                    job_concepts="\t".join(concepts)
                )
            }
        ],
        tools=[initial_schema],
        tool_choice={"type": "function", "function": {"name": "Information-Parser"}},
        temperature=1,
    )
    return eval(first_response.choices[0].message.tool_calls[0].function.arguments)

def fix_duplicate(gen_client : OpenAI, llm_id : str, questions : list[str], concepts : list[str]):
    response = gen_client.chat.completions.create(
        model=llm_id,
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": initial_prompt_template.format(
                    prev_questions="\n".join(questions),
                    job_concepts="\t".join(concepts)
                )
            }
        ],
        tools=[initial_schema],
        tool_choice={"type": "function", "function": {"name": "Information-Parser"}},
        temperature=1,
    )
    return eval(response.choices[0].message.tool_calls[0].function.arguments)

def general_question(gen_client : OpenAI, llm_id : str, questions : list[str], concepts : list[str], concept_asked : str, question : str, answer : str):
    response = gen_client.chat.completions.create(
        model=llm_id,
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": prompt_template.format(
                    prev_questions="\n".join(questions),
                    context=f"Concept : {concept_asked}\n Question : {question}\n Answer : {answer}\n",
                    job_concepts="\t".join(concepts)
                )
            }
        ],
        tools=[general_schema],
        tool_choice={"type": "function", "function": {"name": "InterviewEvaluation"}},
        temperature=1,
    )
    parsed_response = response.choices[0].message
    if parsed_response.content is not None:
         parsed_response = parsed_response.content
         match = re.search(r'"parameters": (\{.*?\})', parsed_response)
         json_match = re.search(r'\{\s*"name":\s*"InterviewEvaluation".*?\}\s*\}', parsed_response)
         if match:
            parameters_json = match.group(1)
            parameters_dict = json.loads(parameters_json)
            return parameters_dict
         elif json_match:
            json_string = json_match.group(0)
            json_string = re.sub(r',\s*}', '}', json_string)
            json_string = re.sub(r',\s*]', ']', json_string)
            parsed_json = json.loads(json_string)
            parameters = parsed_json.get("parameters", {})
            return parameters
         else:
            print("No match found")
            print(parsed_response)
    else:
         parsed_response = parsed_response.tool_calls[0].function.arguments
         parameters_dict = eval(parsed_response)
         return parameters_dict