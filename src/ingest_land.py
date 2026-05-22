import re
import pypdf
import pandas as pd

def process_land_law(pdf_path, act_label, skip_pages, output_path):
    print(f"Opening file: {pdf_path} ({act_label})...")
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    
    # Step 1: Skip the Index / Table of Contents pages
    # range(skip_pages, len(reader.pages)) ensures processing starts exactly on the body text
    for page_num in range(skip_pages, len(reader.pages)):
        text = reader.pages[page_num].extract_text()
        
        # Strip out repetitive running page headers
        text = re.sub(r"THE TRANSFER OF PROPERTY ACT, 1882.*?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"THE REGISTRATION ACT, 1908.*?\n", "", text, flags=re.IGNORECASE)
        
        full_text += text + "\n"
        
    # Step 2: Boundary Regex Extraction
    # Matches a newline (\n) followed by a section number (\d+) and a literal period (\.)
    # Example match: "\n54. " -> group(1) isolates integer 54
    boundary_pattern = r"\n(\d+)\.\s+"
    matches = list(re.finditer(boundary_pattern, full_text))
    parsed_records = []
    
    print(f"Slicing text blocks based on boundaries... Total matched structural indices: {len(matches)}")
    for i in range(len(matches)):
        start_idx = matches[i].start()
        # The window ends where the next section starts, or at the end of the text stream
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        section_id = int(matches[i].group(1))
        raw_content = full_text[start_idx:end_idx].strip()
        
        # Step 3: Flatten internal layout line breaks for clean RAG context windows
        clean_content = re.sub(r'\n+', ' ', raw_content)
        
        parsed_records.append({
            "act": act_label,
            "section_id": section_id,
            "text_content": clean_content
        })
        
    # Step 4: Structuring, Deduplication, and Output Aggregation
    df = pd.DataFrame(parsed_records)
    df['text_len'] = df['text_content'].str.len()
    
    # Group by section_id and pick the row with the longest text content
    # This filters out cross-reference noise in schedules and keeps the primary definition block
    df_clean = df.loc[df.groupby('section_id')['text_len'].idxmax()].sort_values('section_id')
    
    df_clean[['act', 'section_id', 'text_content']].to_csv(output_path, index=False)
    print(f"Successfully finalized {act_label}! Saved {len(df_clean)} sections to {output_path}\n")
    return df_clean

if __name__ == "__main__":
    # Process the Core Transfer of Property Act (Skip first 6 pages of index)
    process_land_law(
        pdf_path="data/TransferProperty1882.pdf", 
        act_label="PROPERTY_1882", 
        skip_pages=6, 
        output_path="data/property_transfer_cleaned.csv"
    )
    
    # Process the supporting Registration Act (Skip first 3 pages of index)
    process_land_law(
        pdf_path="data/Registration1908.pdf", 
        act_label="REGISTRATION_1908", 
        skip_pages=3, 
        output_path="data/property_registration_cleaned.csv"
    )