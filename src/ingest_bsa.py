import re
import pypdf
import pandas as pd

def clean_bsa_pdf(pdf_path="data/BSA_2023.pdf", output_path="data/bsa_cleaned.csv"):
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    
    # Process every page (No initial index tables to skip)
    for page in reader.pages:
        text = page.extract_text()
        
        # Strip recurring official header/footer stamps
        text = re.sub(r"THE GAZETTE OF INDIA EXTRAORDINARY.*?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"SEC\. 1\]", "", text, flags=re.IGNORECASE)
        
        full_text += text + "\n"
        
    # Boundary split strategy: Find line breaks followed by a number and period
    boundary_pattern = r"\n(\d+)\."
    matches = list(re.finditer(boundary_pattern, full_text))
    parsed_sections = []
    
    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        section_id = int(matches[i].group(1))
        raw_content = full_text[start_idx:end_idx].strip()
        
        # Flatten column lines for clean RAG reading
        clean_content = re.sub(r'\n+', ' ', raw_content)
        
        parsed_sections.append({
            "act": "BSA_2023",
            "section_id": section_id,
            "text_content": clean_content
        })
        
    df = pd.DataFrame(parsed_sections)
    # Deduplicate and capture the primary full section descriptions
    df['text_len'] = df['text_content'].str.len()
    df = df.loc[df.groupby('section_id')['text_len'].idxmax()].sort_values('section_id')
    
    df[['act', 'section_id', 'text_content']].to_csv(output_path, index=False)
    print(f"Successfully processed BSA! Saved {len(df)} pristine sections to {output_path}")
    return df

if __name__ == "__main__":
    clean_bsa_pdf()