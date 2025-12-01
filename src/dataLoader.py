from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    Docx2txtLoader)
import tempfile
import os

def load_documents(uploaded_file):
    """
    Data Loadder for all type of uploaded Documents PPTX,Docs,TXT,PDF.
    Return a list of langchain Document object.
    """
    docs=[]

    for file in uploaded_file:

        file_name=file.name.lower()
        with tempfile.NamedTemporaryFile(delete=False,suffix=file_name) as tmp:
            tmp.write(file.read())
            tmp_path=tmp.name
        
        if tmp_path.endswith(".pdf"):
            try:
                loader=PyMuPDFLoader(tmp_path)
            except Exception as e:
                print(f"[ERROR] Failed to load pdf {tmp_path}: {e}")

        
        elif tmp_path.endswith(".docx"):
            try:
                loader=Docx2txtLoader(tmp_path)
            except Exception as e:
                print(f"[ERROR] Failed to load docx {tmp_path}: {e}")

        elif tmp_path.endswith(".pptx"):
            try:
                loader=UnstructuredPowerPointLoader(tmp_path)
            except Exception as e:
                print(f"[ERROR] Failed to load pptx {tmp_path}: {e}")
        
        elif tmp_path.endswith(".txt"):
            try:
                loader=TextLoader(tmp_path)
            except Exception as e:
                print(f"[ERROR] Failed to load txt {tmp_path}: {e}")

        else:
            print(f"unsupported file {file_name}")
            continue

        file_docs=loader.load()

        docs.extend(file_docs)

        os.remove(tmp_path)

    print(f"Retuning the docs length: {len(docs)}")
    
    return docs

if __name__=="__main__":
    import glob
    sample_files = [
       open(r"Rich-Dad-Poor-Dad.pdf","rb"),
       open(r"journal_editingfinal.docx","rb"),
       open(r"Project-PPT-model.pptx","rb"),
       open(r"healthcare_report.txt","rb")
    ]
    docs=load_documents(sample_files)
    print(f"the lenght of the docs {len(docs)}")
