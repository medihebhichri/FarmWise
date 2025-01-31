#!/bin/bash

# Define the destination folder
DEST_FOLDER="Organic_Eprints_Meta_Data"

# Create the folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

# Find and copy all .json files to the destination folder
find . -type f -name "*.json" -exec cp {} "$DEST_FOLDER"/ \;

echo "✅ All .json files have been copied to $DEST_FOLDER."
