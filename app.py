import streamlit as st

st.title("🎓 Student Result Analyzer")

name = st.text_input("Enter Student Name")
roll = st.text_input("Enter Roll Number")

st.subheader("Enter Marks (Out of 100)")

subject1 = st.number_input("SE Marks", min_value=0, max_value=100)
subject2 = st.number_input("CN Marks", min_value=0, max_value=100)
subject3 = st.number_input("WAD Marks", min_value=0, max_value=100)
subject4 = st.number_input("OB Marks", min_value=0, max_value=100)

if st.button("Calculate Result"):
    total = subject1 + subject2 + subject3 + subject4
    percentage = total / 4

    st.success(f"Student Name: {name}")
    st.info(f"Roll Number: {roll}")
    st.write("Total Marks:", total)
    st.write("Percentage:", percentage)

    if percentage >= 90:
        st.success("Grade: A+")
    elif percentage >= 75:
        st.success("Grade: A")
    elif percentage >= 60:
        st.success("Grade: B")
    elif percentage >= 50:
        st.warning("Grade: C")
    else:
        st.error("Grade: Fail")
