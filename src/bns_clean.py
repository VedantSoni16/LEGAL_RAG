import pandas as pd

# 1. Load your raw output file
df = pd.read_csv("data/bns_cleaned.csv")

# 2. Add text length tracking column
df['text_len'] = df['text_content'].str.len()

# 3. Separate the real BNS sections (1 to 358)
# We find the longest text row for each section_id to ensure we get the full text body, not index titles
idx_max_len = df[df['section_id'] <= 358].groupby('section_id')['text_len'].idxmax()
bns_final_clean = df.loc[idx_max_len].sort_values(by='section_id')

# 4. Save your pristine RAG-ready dataset
bns_final_clean[['act', 'section_id', 'text_content']].to_csv("data/bns_final_clean.csv", index=False)
print(f"Pristine dataset saved with exactly {len(bns_final_clean)} sections!")