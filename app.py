from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr
import re

model_link = "text_summary_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_link)
tokenizer = AutoTokenizer.from_pretrained(model_link)

pipe = pipeline('summarization', model=model, tokenizer=tokenizer)
gen_kwargs = {'length_penalty': 0.8, 'num_beams': 8, "min_length": 30}


def dummy_summarize(text):
    text = clean_text(text)
    return pipe(text, **gen_kwargs)[0]['summary_text']


def clean_text(text):
    # Remove Byte Order Marks (BOM)
    text = text.replace('\ufeff', '')
    # Replace \n with a space
    text = text.replace('\n', ' ')
    # Replace \r with a space
    text = text.replace('\r', ' ')
    # Remove backslashes
    text = text.replace('\\', '')
    # Remove Non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove Non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


# Custom HTML and CSS for the title and theme
title_html = """
<h1 style="font-size: 40px; text-align: center; color: white;">Text Summarization</h1>
<p style="text-align: center; color: white; font-size:20px">Enter text to summarize it using a pretrained model.</p>
"""

css = """
body {
    background-color: #007BFF;
    color: white;
}

.gr-textbox textarea {
    background-color: #0056b3;
    color: white;
    border: 2px solid white;
}

.gr-button {
    background-color: #0056b3;
    color: white;
    border: 2px solid white;
}

.gr-button:hover {
    background-color: #004080;
}

.gr-textbox input {
    background-color: #0056b3;
    color: white;
    border: 2px solid white;
}
"""

# Create a Gradio interface with large input and output textboxes
interface = gr.Interface(
    fn=dummy_summarize,
    inputs=gr.Textbox(lines=15, placeholder="Enter text here..."),
    outputs=gr.Textbox(lines=15, placeholder="Summary will appear here..."),
    description=title_html,
    theme="compact",  # Use compact theme to reduce padding
    css=css
)

# Launch the interface
interface.launch()
