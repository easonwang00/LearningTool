#streamlit example
"""
def retrieve_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# create a streamlit app
st.title("Canadian Legal Explainer (that does not give advice)")

if "LegalExpert" not in st.session_state:
    st.session_state.LegalExpert = LegalExpert()

# create a upload file widget for a pdf
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if a pdf file is uploaded
if pdf_file:
    # retrieve the text from the pdf
    if "context" not in st.session_state:
        st.session_state.context = retrieve_pdf_text(pdf_file)



# create a button that clears the context
if st.button("Clear context"):
    st.session_state.__delitem__("context")
    st.session_state.__delitem__("legal_response")

# if there's context, proceed
if "context" in st.session_state:
    # create a dropdown widget for the language
    language = st.selectbox("Language", ["English", "Français"])
    # create a text input widget for a question
    question = st.text_input("Ask a question")

    # create a button to run the model
    if st.button("Run"):
        # run the model
        legal_response = st.session_state.LegalExpert.run_chain(
            language=language, context=st.session_state.context, question=question
        )

        if "legal_response" not in st.session_state:
            st.session_state.legal_response = legal_response

        else:
            st.session_state.legal_response = legal_response

# display the response
if "legal_response" in st.session_state:
    st.write(st.session_state.legal_response)

"""