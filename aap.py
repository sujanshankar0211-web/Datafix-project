import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# -----------------------------------------------------------------------------
# 1. PAGE & APPLICATION CONFIGURATION
# -----------------------------------------------------------------------------
# This should be the first Streamlit command in your app
st.set_page_config(
    page_title="DataFix Enterprise | Autonomous Data Quality Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
# Initialize session state variables to avoid errors on first run
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "changes_summary" not in st.session_state:
    st.session_state.changes_summary = {}
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# -----------------------------------------------------------------------------
# 3. CORE HELPER FUNCTIONS (Modules)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_file):
    """
    Loads data from an uploaded file into a pandas DataFrame.
    Includes robust error handling for parsing issues.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."

        if df.empty:
            return None, "The uploaded file is empty. Please upload a valid file."

        return df, None

    except pd.errors.ParserError:
        return None, "Error parsing the file. Please ensure it is a well-formatted file."
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def handle_missing_values(df: pd.DataFrame, method: str, columns: list = None):
    """Handles missing values in the DataFrame."""
    df_copy = df.copy()
    if method == "Drop Rows":
        return df_copy.dropna(subset=columns)
    
    for col in columns:
        if method == "Fill with Mean":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                fill_value = df_copy[col].mean()
                df_copy[col] = df_copy[col].fillna(fill_value)
        elif method == "Fill with Median":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                fill_value = df_copy[col].median()
                df_copy[col] = df_copy[col].fillna(fill_value)
        elif method == "Fill with Mode":
            fill_value = df_copy[col].mode()
            if not fill_value.empty:
                df_copy[col] = df_copy[col].fillna(fill_value.iloc[0])
    return df_copy

def remove_duplicates(df: pd.DataFrame):
    """Removes duplicate rows from the DataFrame."""
    return df.drop_duplicates()

def get_comparison_summary(original_df, cleaned_df):
    """Generates a summary of changes between two DataFrames."""
    if original_df is None or cleaned_df is None:
        return {}
        
    summary = {
        "original_rows": len(original_df),
        "cleaned_rows": len(cleaned_df),
        "rows_removed": len(original_df) - len(cleaned_df),
        "original_cols": len(original_df.columns),
        "cleaned_cols": len(cleaned_df.columns),
        "original_missing": int(original_df.isnull().sum().sum()),
        "cleaned_missing": int(cleaned_df.isnull().sum().sum()),
        "duplicates_removed": len(original_df) - len(original_df.drop_duplicates()),
    }
    summary["missing_fixed"] = summary["original_missing"] - summary["cleaned_missing"]
    return summary

def style_comparison_df(original_df, cleaned_df):
    """
    Compares two DataFrames and returns a styled DataFrame highlighting changes.
    """
    # Align columns for comparison, ensuring no data loss on column changes
    original_aligned, cleaned_aligned = original_df.align(cleaned_df, join='outer', axis=1, fill_value=np.nan)

    # Use pandas compare to find cell-level differences
    diff = original_aligned.compare(cleaned_aligned, align_axis=1, keep_shape=True, keep_equal=True)

    def highlight_changes(s):
        is_na = pd.isna(s)
        return ['' if v else 'background-color: #FFF3CD' for v in is_na]

    # Apply styling to highlight changes
    # This is a simplified approach; for complex changes, more advanced styling is needed.
    styled_df = diff.style.highlight_null(color='#F8D7DA').highlight_not_null(color='#D4EDDA')
    
    return styled_df


# -----------------------------------------------------------------------------
# 4. SIDEBAR UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ DataFix Studio")
    st.markdown("Autonomous Data Cleaning & Quality Assurance")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Raw Dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )
    
    if uploaded_file:
        # Check if it's a new file
        if st.session_state.last_uploaded_file != uploaded_file.name:
            df, error = load_data(uploaded_file)
            if error:
                st.error(error)
                st.session_state.raw_df = None
                st.session_state.cleaned_df = None
            else:
                st.session_state.raw_df = df
                st.session_state.cleaned_df = df.copy()
                st.session_state.last_uploaded_file = uploaded_file.name
                st.success("Dataset loaded!")
    
    st.divider()
    
    if st.session_state.get("cleaned_df") is not None:
        st.header("Download Results")
        
        # Create an in-memory Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.cleaned_df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
            if st.session_state.raw_df is not None:
                st.session_state.raw_df.to_excel(writer, sheet_name='Original_Data', index=False)
        
        st.download_button(
            label="Download Cleaned Excel File",
            data=output.getvalue(),
            file_name='cleaned_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

    st.divider()
    st.caption("Engineered for enterprise quality.")


# -----------------------------------------------------------------------------
# 5. MAIN WORKFLOW & UI
# -----------------------------------------------------------------------------

# --- Main App Title ---
st.title("Data Cleaning & Comparison Toolkit")

# --- Check for Uploaded Data ---
if st.session_state.get("cleaned_df") is None:
    st.info("Please upload a dataset using the sidebar to get started.")
    st.stop()

# --- DataFrames to use ---
raw_df = st.session_state.raw_df
cleaned_df = st.session_state.cleaned_df

# --- Top-level Metrics ---
st.header("Data Quality Dashboard")
summary = get_comparison_summary(raw_df, cleaned_df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{summary.get('cleaned_rows', 0)}", f"{summary.get('rows_removed', 0)} Removed")
col2.metric("Columns", f"{summary.get('cleaned_cols', 0)}")
col3.metric("Missing Values", f"{summary.get('cleaned_missing', 0)}", f"{summary.get('missing_fixed', 0)} Fixed")
col4.metric("Duplicate Rows", f"{cleaned_df.duplicated().sum()}")

st.divider()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["Data Cleaning Toolkit", "Before & After Comparison", "Raw Data Preview"])

with tab1:
    st.header("Cleaning Toolkit")
    
    # --- Handling Missing Values ---
    with st.expander("Handle Missing Values", expanded=True):
        missing_summary = cleaned_df.isnull().sum()
        cols_with_missing = missing_summary[missing_summary > 0].index.tolist()
        
        if not cols_with_missing:
            st.success("No missing values found in the current dataset.")
        else:
            st.write("Columns with missing values:")
            st.dataframe(missing_summary[missing_summary > 0].to_frame(name='Missing Count'), use_container_width=True)

            method = st.selectbox(
                "Choose a method to handle missing values:",
                ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"]
            )
            
            selected_cols = st.multiselect(
                "Select columns to apply the method to:",
                options=cols_with_missing,
                default=cols_with_missing
            )

            if st.button("Apply Missing Value Fix", type="primary"):
                if selected_cols:
                    st.session_state.cleaned_df = handle_missing_values(cleaned_df, method, selected_cols)
                    st.success(f"Applied '{method}' to columns: {', '.join(selected_cols)}")
                    st.rerun()
                else:
                    st.warning("Please select at least one column.")

    # --- Removing Duplicates ---
    with st.expander("Remove Duplicate Rows"):
        num_duplicates = cleaned_df.duplicated().sum()
        if num_duplicates == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"Found {num_duplicates} duplicate rows.")
            if st.button("Remove All Duplicate Rows", type="primary"):
                st.session_state.cleaned_df = remove_duplicates(cleaned_df)
                st.success("Duplicate rows have been removed.")
                st.rerun()

with tab2:
    st.header("Before & After Comparison")
    st.info("This view highlights the changes between your original and cleaned datasets.")
    
    st.markdown("""
    - <span style='background-color: #D4EDDA'>Green</span>: Value was filled or is unchanged.
    - <span style='background-color: #F8D7DA'>Red</span>: Value was removed (is now empty).
    """, unsafe_allow_html=True)

    # Generate and display the styled comparison DataFrame
    if st.button("Generate Comparison View"):
        with st.spinner("Comparing datasets..."):
            styled_comparison = style_comparison_df(raw_df, cleaned_df)
            st.write(styled_comparison.to_html(), unsafe_allow_html=True)

with tab3:
    st.header("Current Dataset Preview")
    st.dataframe(cleaned_df, use_container_width=True)

