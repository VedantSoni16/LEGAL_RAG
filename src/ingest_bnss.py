import re
import pypdf
import pandas as pd

def clean_bnss_pdf_fixed(pdf_path="data/Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf", output_path="data/bnss_final_clean.csv"):
    print("Opening BNSS Parliament Bill...")
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    
    # Core Strategy: Skip the first 17 pages of index tables entirely. Start processing from Page 18.
    for page_num in range(17, len(reader.pages)):
        text = reader.pages[page_num].extract_text()
        
        # Wipe page layout artifact lines
        text = re.sub(r"ARRANGEMENT OF CLAUSES.*?\n", "", text, flags=re.IGNORECASE)
        full_text += text + "\n"
        
    # Split on Clause markers starting on a newline followed by number and period
    boundary_pattern = r"\n(\d+)\.\s+"
    matches = list(re.finditer(boundary_pattern, full_text))
    parsed_clauses = []
    
    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        clause_id = int(matches[i].group(1))
        
        # Only extract actual legal sections (clauses 1 to 531)
        if clause_id > 531:
            continue
            
        raw_content = full_text[start_idx:end_idx].strip()
        clean_content = re.sub(r'\n+', ' ', raw_content)
        
        parsed_clauses.append({
            "act": "BNSS_2023",
            "section_id": clause_id,
            "text_content": clean_content
        })
        
    df = pd.DataFrame(parsed_clauses)
    df['text_len'] = df['text_content'].str.len()
    
    # Max length aggregation filters out miscellaneous cross-references in schedules
    df_clean = df.loc[df.groupby('section_id')['text_len'].idxmax()].sort_values('section_id')
    
    df_clean[['act', 'section_id', 'text_content']].to_csv(output_path, index=False)
    print(f"Successfully processed fixed BNSS! Saved {len(df_clean)} pristine sections to {output_path}")
    return df_clean

if __name__ == "__main__":
    clean_bnss_pdf_fixed()