import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/validate"

st.set_page_config(page_title="AI Prescription Validation", layout="wide")

st.title("AI Clinical Decision Support System")
st.markdown("### Prescription Appropriateness Validator")

st.divider()

col1, col2 = st.columns(2)

with col1:
    disease = st.text_input("Enter Disease")

with col2:
    drugs = st.text_input("Enter Drugs (comma separated)")

st.divider()

if st.button("Validate Prescription"):
    if not disease or not drugs:
        st.warning("Please enter both disease and drugs.")
    else:
        drug_list = [d.strip() for d in drugs.split(",") if d.strip()]

        if not drug_list:
            st.warning("Please provide at least one valid drug name.")
        else:
            with st.spinner("Analyzing prescription..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"disease": disease, "drugs": drug_list},
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.subheader("Analysis Result")

                    for item in result.get("analysis", []):
                        prediction = str(item.get("prediction", "")).lower()
                        drug = item.get("drug", "Unknown drug")

                        if prediction == "appropriate":
                            st.success(f"{drug} -> Appropriate for {disease}")
                        else:
                            st.error(f"{drug} -> Not appropriate for {disease}")

                    st.divider()

                    final_verdict = result.get("final_verdict", "No verdict returned.")
                    if "appropriate" in final_verdict.lower() and "not" not in final_verdict.lower():
                        st.success(final_verdict)
                    else:
                        st.error(final_verdict)

                except requests.exceptions.ConnectionError:
                    st.error("Backend API not running. Please start FastAPI.")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Please try again.")
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Backend returned an error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
