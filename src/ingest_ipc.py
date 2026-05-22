import re
import fitz  # PyMuPDF
import pandas as pd

def process_ipc_textbook(pdf_path, output_csv_path="data/ipc_cleaned.csv"):
    print("Opening giant IPC commentary book...")
    doc = fitz.open(pdf_path)
    full_text = ""
    
    # Step 1: Scan pages to build text corpus
    print("Extracting text from pages...")
    for page in doc:
        text = page.get_text("text")
        full_text += text + "\n"
        
    # Step 2: Regex setup to track structural markers
    # Pattern looks for literal '[s ', then captures the digits, followed by ']'
    # Example match: "[s 22]" -> group(1) isolates "22"
    section_pattern = r"\[s\s+(\d+)\]"
    
    matches = list(re.finditer(section_pattern, full_text))
    parsed_ipc_sections = []
    
    print(f"Parsing matches... Found potential structural hits: {len(matches)}")
    
    for i in range(len(matches)):
        start_idx = matches[i].start()
        # Bound the slice to the next identified section marker
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        section_id = matches[i].group(1)
        raw_chunk = full_text[start_idx:end_idx].strip()
        
        # Keep chunks meaningful. If a section chunk is tiny or a fluke, ignore it.
        # But if it contains legal code or commentary, clean up text breaks.
        if len(raw_chunk) > 30:
            # Flatten internal narrow-column text linebreaks into one line
            clean_chunk = re.sub(r'\n+', ' ', raw_chunk)
            
            # Extract just the direct text. To prevent overloading the chunk with 10 pages of
            # commentary, we can optionally cap the max context window to the first 1200 characters.
            truncated_chunk = clean_chunk[:1200]
            
            parsed_ipc_sections.append({
                "act": "IPC_1860",
                "section_id": int(section_id),
                "text_content": truncated_chunk
            })
            
    # Step 3: De-duplicate and align structure
    df = pd.DataFrame(parsed_ipc_sections)
    
    # Sort numerically and keep the deepest entry description
    df = df.sort_values(by=["section_id", "text_content"], ascending=[True, False])
    df = df.drop_duplicates(subset=["section_id"], keep="first")
    
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully processed {len(df)} core IPC sections! Saved to {output_csv_path}")
    return df

if __name__ == "__main__":
    # Point this to your uploaded file path
    df_ipc = process_ipc_textbook("data\Indian20Code%20Book.pdf", "data/ipc_cleaned.csv")
    print(df_ipc.head())