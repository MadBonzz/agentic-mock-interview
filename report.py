import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Spacer
from openai import OpenAI

def get_df(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parsed_line = eval(line.strip())
            if isinstance(parsed_line, dict):
                data.append(parsed_line)
            
    rows = []
    for i in range(1, len(data)):
        if i == 1:
            concept = data[i - 1].get('concept_asked', None)
        else:
            concept = data[i].get('new_concept', None)
        previous_category = data[i - 1].get('question_category', None) 

        current_review = {
            'Category': previous_category,
            'Concept' : concept,
            'Evaluation' : f"Technical : {data[i].get('technical_ability', None)} Language : {data[i].get('language_clarity', None)} Knowledge : {data[i].get('depth_of_knowledge', None)}",
            'Review' : data[i].get('review', None),
        }
        rows.append(current_review)
    df = pd.DataFrame(rows)
    return df

def get_review(df : pd.DataFrame):
    review_string = ""
    for idx, row in df.iterrows():
        curr_row = ""
        for col in df.columns:
            curr_row += f"{col} : {row[col]}\t"
        curr_row += "\n"
        review_string += curr_row
    return review_string

prompt_template = """
Based on this detailed evaluation provided for an interview conducted for a user, provide an overall review for the user.
Analyse the concepts, categories, Evaluation and question wise reveiw properly to generate a comprehensive overall review for the user
inclusive of strengths, weakness and concepts to focus on. Always start the review with the word Review.

Detailed Evaluation : {eval}

Review : 
"""


def overall_review(gen_client : OpenAI, llm_id : str, review_string : str):
    response = gen_client.chat.completions.create(
        model=llm_id,
        messages=[
            {"role" : "system",
             "content" : "You are an AI evaluator. You will be provided by a detailed interview evaluation by the user. Create an overall review using it as instructed by the user."
            },
            {
                "role": "user",
                "content": prompt_template.format(
                    eval=review_string
                )
            }
        ],
        temperature=1,
    )
    return response.choices[0].message.content

def generate_pdf(df : pd.DataFrame, output_path, final_review):
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]

    wrapped_data = [[Paragraph(str(cell), styleN) for cell in row] for row in [df.columns.tolist()] + df.values.tolist()]

    doc = SimpleDocTemplate(output_path, pagesize=letter)

    col_widths = [80, 100, 90, 250]

    table = Table(wrapped_data, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    table_heading = Paragraph("<b>Detailed Evaluation</b>", styles['Heading2'])
    final_review_heading = Paragraph("<b>Final Review</b>", styles["Heading2"])
    final_review = final_review.replace("\n", "<br/>")
    final_review = final_review.replace("**", "")
    final_review_paragraph = Paragraph(final_review, styleN)

    spacer = Spacer(1, 12)

    doc.build([table_heading, spacer, table, spacer, final_review_heading, spacer, final_review_paragraph])
