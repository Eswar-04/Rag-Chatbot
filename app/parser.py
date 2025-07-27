from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to load a PDF and split its content into smaller text chunks
def load_and_split_pdfs(pdf_path="data/Artificial Intelligence.pdf"):
    # Load the PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Load all pages as Document objects

    # Initialize the text splitter with chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Each chunk will have up to 500 characters
        chunk_overlap=50     # Consecutive chunks will overlap by 50 characters
    )

    # Split the loaded documents into manageable chunks
    chunks = splitter.split_documents(documents)

    # Return the list of split chunks
    return chunks
