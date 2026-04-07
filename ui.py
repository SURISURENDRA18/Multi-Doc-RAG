import streamlit as st
import requests

#Correct port
API_URL = "http://127.0.0.1:8002/ask"

st.set_page_config(page_title="Multi-Doc RAG", layout="wide")

st.title("Multi-Doc RAG System")

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={"question": query},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Safe access
                answer = data.get("answer", "No answer returned")
                sources = data.get("sources", [])

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources")
                if sources:
                    for src in sources:
                        st.write(f"- {src}")
                else:
                    st.write("No sources found")

            else:
                st.error(f"API error: {response.status_code}")

        except Exception as e:
            st.error(f"Connection failed: {e}")
