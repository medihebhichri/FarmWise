import fitz  # PyMuPDF
import os
from tqdm import tqdm

def extract_text_from_pdfs(folder_path, output_file="agriculture_knowledge.txt"):
    all_text = ""

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            print(f"📘 Processing: {filename}")
            file_path = os.path.join(folder_path, filename)
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text()
                all_text += text + "\n"
            doc.close()

    if all_text.strip() == "":
        print("⚠️ No text was extracted. The PDFs may be scanned images.")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"\n✅ Text saved to: {output_file}")

# ✅ Use raw string (r"...") to avoid escape issues
extract_text_from_pdfs(r"C:\Users\msi\Desktop\FINAL\BOOKS")
