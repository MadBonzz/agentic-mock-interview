from openai import OpenAI
from generate_response import extraction, initial_question, general_question
from report import get_df, get_review, overall_review, generate_pdf

llm_id = 'meta-llama-3.1-8b-instruct'
server_url = 'http://127.0.0.1:1234/v1'
review_path = "output.txt"
report_path = "output.pdf"

gen_client = OpenAI(base_url=server_url, api_key="lm-studio")

df = get_df(review_path)
eval_string = get_review(df)
final_review = overall_review(gen_client, llm_id, eval_string)
print(final_review)
generate_pdf(df, report_path, final_review)