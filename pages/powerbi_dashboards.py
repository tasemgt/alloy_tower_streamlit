import streamlit as st

def show_powerbi_dashboard():
    st.title("Business Intelligence Dashboard")

    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiZjYwNDRhOTMtOTRmZC00Mzk2LWJkN2EtM2Q4ZDJkODViOTAyIiwidCI6ImZmMGYzZTNhLTNlNTMtNDU0Zi1iMmI1LTZjNjg3NTNiOGVlNCJ9"

    # st.markdown("""
    #     <style>
    #     iframe {
    #         transform: scale(0.8);
    #         transform-origin: 0 0;
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)
    st.components.v1.iframe(
        powerbi_url,
        height=950,
        width=1500,
        scrolling=True
    )

show_powerbi_dashboard()
