import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="DATAFIX",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 DATAFIX")
st.subheader("Autonomous Data Quality and Repair Engine")

st.write(
    "Upload your dataset, detect data quality problems, "
    "automatically repair them and download clean data."
)

# --------------------------------
# FILE UPLOAD
# --------------------------------

st.header("1. Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Choose CSV or Excel file",
    type=["csv", "xlsx"]
)

# --------------------------------
# DATA CLEANING FUNCTION
# --------------------------------

def clean_dataset(data, fill_missing, remove_duplicates,
                  normalize_city, validate_data):

    df = data.copy()

    original_rows = len(df)

    # Remove extra spaces from text
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # Standardize city names
    if normalize_city:
        for col in df.columns:
            if "city" in col.lower():
                df[col] = df[col].astype("string").str.strip().str.title()

                city_mapping = {
                    "Bengaluru": "Bengaluru",
                    "Bangalore": "Bengaluru",
                    "Mysore": "Mysuru",
                    "Mysuru": "Mysuru"
                }

                df[col] = df[col].replace(city_mapping)

    # Validate age
    if validate_data:
        for col in df.columns:
            if "age" in col.lower():
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

                df.loc[
                    (df[col] < 0) | (df[col] > 120), col
                ] = np.nan

    # Validate email
    if validate_data:
        for col in df.columns:
            if "email" in col.lower():
                email_pattern = r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$"

                valid = df[col].astype("string").str.match(
                    email_pattern, na=False
                )

                df.loc[~valid, col] = pd.NA

    # Validate date columns
    if validate_data:
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(
                    df[col], errors="coerce"
                )

    # Remove duplicate rows
    duplicates_found = int(df.duplicated().sum())

    if remove_duplicates:
        df = df.drop_duplicates()

    # Fill missing values
    if fill_missing:

        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].notna().any():
                    df[col] = df[col].fillna(df[col].median())

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Keep missing dates blank rather than inventing dates
                pass

            else:
                # Do not invent email addresses for invalid/missing emails
                if "email" in col.lower():
                    continue

                mode_value = df[col].mode(dropna=True)

                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value.iloc[0])

    repaired_rows = len(df)

    return df, original_rows, repaired_rows, duplicates_found


# --------------------------------
# MAIN APPLICATION
# --------------------------------

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            original_df = pd.read_csv(uploaded_file)
        else:
            original_df = pd.read_excel(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # Original dataset
        st.header("2. Original Dataset Preview")

        st.dataframe(
            original_df.head(10),
            use_container_width=True
        )

        # Dataset information
        st.header("3. Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", original_df.shape[0])

        with col2:
            st.metric("Total Columns", original_df.shape[1])

        with col3:
            st.metric(
                "Missing Cells",
                int(original_df.isnull().sum().sum())
            )

        # Missing values
        st.subheader("Missing Values Report")

        missing_report = pd.DataFrame({
            "Column": original_df.columns,
            "Missing Values": original_df.isnull().sum().values
        })

        st.dataframe(
            missing_report,
            use_container_width=True
        )

        # Duplicate report
        st.subheader("Duplicate Records")

        duplicate_count = int(original_df.duplicated().sum())

        st.write(
            f"Total duplicate rows detected: {duplicate_count}"
        )

        # --------------------------------
        # CLEANING OPTIONS
        # --------------------------------

        st.header("4. Automatic Data Cleaning")

        st.write("Select the cleaning operations you want to perform.")

        fill_missing = st.checkbox(
            "Fill missing numeric values with median and other eligible missing values with mode",
            value=True
        )

        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=True
        )

        normalize_city = st.checkbox(
            "Standardize city names",
            value=True
        )

        validate_data = st.checkbox(
            "Validate age, email and date columns",
            value=True
        )

        # --------------------------------
        # RUN CLEANING
        # --------------------------------

        cleaned_df, old_rows, new_rows, duplicates_found = clean_dataset(
            original_df,
            fill_missing,
            remove_duplicates,
            normalize_city,
            validate_data
        )

        st.success("Data cleaning preview generated!")

        # --------------------------------
        # BEFORE AND AFTER
        # --------------------------------

        st.header("5. Data Quality Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Original Missing Cells",
                int(original_df.isnull().sum().sum())
            )

        with col2:
            st.metric(
                "Remaining Missing Cells",
                int(cleaned_df.isnull().sum().sum())
            )

        with col3:
            st.metric(
                "Duplicate Rows Found",
                duplicates_found
            )

        st.write(f"Original rows: {old_rows}")
        st.write(f"Rows after cleaning: {new_rows}")

        st.subheader("Cleaned Dataset Preview")

        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True
        )

        # --------------------------------
        # DOWNLOAD EXCEL
        # --------------------------------

        st.header("6. Download Cleaned Dataset")

        excel_buffer = BytesIO()

        report = pd.DataFrame({
            "Metric": [
                "Original Rows",
                "Cleaned Rows",
                "Original Columns",
                "Original Missing Cells",
                "Remaining Missing Cells",
                "Duplicate Rows Found"
            ],
            "Value": [
                old_rows,
                new_rows,
                original_df.shape[1],
                int(original_df.isnull().sum().sum()),
                int(cleaned_df.isnull().sum().sum()),
                duplicates_found
            ]
        })

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            cleaned_df.to_excel(
                writer,
                sheet_name="Cleaned_Data",
                index=False
            )

            report.to_excel(
                writer,
                sheet_name="Quality_Report",
                index=False
            )

        st.download_button(
            label="📥 Download Cleaned Excel File",
            data=excel_buffer.getvalue(),
            file_name="DATAFIX_Cleaned_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # CSV download
        csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Cleaned CSV File",
            data=csv_data,
            file_name="DATAFIX_Cleaned_Data.csv",
            mime="text/csv"
        )

        st.balloons()

        st.success(
            "DATAFIX processing completed! Download your cleaned dataset above."
        )

    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info("Please upload a CSV or Excel dataset to begin.")

    st.write("DATAFIX features:")
    st.write("- Missing value detection")
    st.write("- Duplicate record detection")
    st.write("- Data validation")
    st.write("- Automatic data cleaning")
    st.write("- Excel and CSV export")