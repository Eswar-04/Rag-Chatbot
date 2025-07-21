import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#  Folder where all your PDF files are kept
DATA_FOLDER = "data"

#  Function to load PDFs and split them into smaller chunks
def load_and_split_pdfs():
    all_chunks = []  #  Store all text chunks here

    #  Loop through every file in the data folder
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_FOLDER, filename)
            print(f"  Loading: {filename}")

            #  Load PDF and convert to raw text using PyPDFLoader
            loader = PyPDFLoader(filepath)
            documents = loader.load()

            #  Split text into smaller chunks for embedding
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,      # size of each chunk
                chunk_overlap=50     # how much chunks overlap
            )
            chunks = splitter.split_documents(documents)  #  do the split
            all_chunks.extend(chunks)  #  Add to final list

    print(f"  Total chunks loaded: {len(all_chunks)}")
    return all_chunks  #  Return all chunks
