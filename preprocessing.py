import re

def clean_text(text: str) -> str:
    # clean html tags
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    return re.sub(r'[*_`#\[\](){}]', ' ', text)  # Remove markdown tags