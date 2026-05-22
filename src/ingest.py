import re
import fitz  # PyMuPDF (Install via: pip install pymupdf)
import pandas as pd

def process_bns_pdf(pdf_path, output_csv_path="bns_cleaned.csv"):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    # Step 1: Skip the Index pages. 
    # In the official Act 45 of 2023, Chapter 1 starts around page 13 or 14 (Index 12/13)
    # We will loop from page 12 onwards to avoid the Table of Contents.
    for page_num in range(12, len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Step 2: Clean up recurring Gazette header noise via basic regex
        text = re.sub(r"THE GAZETTE OF INDIA EXTRAORDINARY.*?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"SEC\. 1\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"PART II—", "", text, flags=re.IGNORECASE)
        
        full_text += text + "\n"
        
    # Step 3: Split the entire corpus by section boundaries
    # Official structure: A line starting with a number, a period, space, and a title
    # Pattern looks for: Newline -> Digit(s) -> Period -> Space
    section_pattern = r"\n(\d+)\.\s+"
    
    matches = list(re.finditer(section_pattern, full_text))
    parsed_sections = []
    
    for i in range(len(matches)):
        start_idx = matches[i].start()
        # End index is where the next section starts, or the end of the text file
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        # Extract the matched Section Number
        section_id = matches[i].group(1)
        
        # Slice out the raw chunk text
        raw_chunk = full_text[start_idx:end_idx].strip()
        
        # Clean up double newlines inside the text chunk for clean rendering
        clean_chunk = re.sub(r'\n+', ' ', raw_chunk)
        
        parsed_sections.append({
            "act": "BNS_2023",
            "section_id": section_id,
            "text_content": clean_chunk
        })
        
    # Step 4: Output to a structured data format
    df = pd.DataFrame(parsed_sections)
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully extracted {len(df)} sections! Saved to {output_csv_path}")
    return df


if __name__ == "__main__":
    df = process_bns_pdf("data/BNS2023.pdf", output_csv_path="data/bns_cleaned.csv")
    print(df.head())