import pandas as pd

# Load your file
df = pd.read_excel("input.xlsx")

# Clean object/string columns
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].where(df[c].isna(), df[c].astype(str))

# Export cleaned file
df.to_excel("cleaned_output.xlsx", index=False)

print("Data cleaning completed successfully!")
