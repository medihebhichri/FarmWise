import os
import bibtexparser
import json

# Define the root directory where .bib files are stored
root_dir = "."

# Iterate through all directories and find .bib files
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".bib"):
            bib_path = os.path.join(subdir, file)
            json_path = os.path.join(subdir, file.replace(".bib", ".json"))

            # Read the .bib file
            with open(bib_path, "r", encoding="utf-8") as bib_file:
                bib_database = bibtexparser.load(bib_file)

            # Convert to JSON
            json_output = json.dumps(bib_database.entries, indent=4, ensure_ascii=False)

            # Save JSON file
            with open(json_path, "w", encoding="utf-8") as json_file:
                json_file.write(json_output)

            print(f"Converted {bib_path} -> {json_path}")

print("All .bib files converted to JSON successfully!")

