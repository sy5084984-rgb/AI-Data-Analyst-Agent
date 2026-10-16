import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ----------------------------------
# APP SETTINGS
# ----------------------------------

st.set_page_config(
    page_title="AI Data Analyst Agent V3",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Analyst Agent — Version 3")
st.write(
    "Upload your CSV or Excel file, explore your data, "
    "create charts, ask questions and download reports."
)

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

def clean_column_name(column):
    return str(column).lower().replace("_", " ").strip()


def find_column(df, keywords, numeric_only=False):
    if numeric_only:
        columns = list(df.select_dtypes(include=np.number).columns)
    else:
        columns = list(df.columns)

    # First try exact match
    for column in columns:
        name = clean_column_name(column)

        if name in keywords:
            return column

    # Then try partial match
    for column in columns:
        name = clean_column_name(column)

        if any(word in name for word in keywords):
            return column

    return None


def make_report(df):
    report = []

    report.append("AI DATA ANALYST AGENT — ANALYSIS REPORT")
    report.append("=" * 45)
    report.append(f"Total rows: {len(df)}")
    report.append(f"Total columns: {len(df.columns)}")
    report.append(f"Duplicate rows: {int(df.duplicated().sum())}")
    report.append(f"Total missing cells: {int(df.isnull().sum().sum())}")
    report.append("")
    report.append("COLUMN INFORMATION")
    report.append("-" * 45)

    for column in df.columns:
        report.append(
            f"{column}: type={df[column].dtype}, "
            f"missing={int(df[column].isnull().sum())}, "
            f"unique={int(df[column].nunique())}"
        )

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        report.append("")
        report.append("NUMERICAL SUMMARY")
        report.append("-" * 45)
        report.append(numeric_df.describe().to_string())

    report.append("")
    report.append("MISSING VALUES")
    report.append("-" * 45)
    report.append(df.isnull().sum().to_string())

    return "\n".join(report)


def analyze_question(df, question):
    q = question.lower().strip()
    q = q.replace("_", " ")

    numeric_df = df.select_dtypes(include=np.number)

    # Row count
    if any(x in q for x in [
        "how many rows", "total rows", "number of rows",
        "kitni rows", "kitne rows"
    ]):
        return f"Dataset mein **{len(df)} rows** hain."

    # Column count
    if any(x in q for x in [
        "how many columns", "total columns",
        "number of columns", "kitne columns"
    ]):
        return f"Dataset mein **{len(df.columns)} columns** hain."

    # Column names
    if any(x in q for x in [
        "column names", "list columns", "show columns",
        "columns ke naam"
    ]):
        return pd.DataFrame({"Column Names": df.columns})

    # Missing values
    if any(x in q for x in [
        "missing", "null values", "missing values",
        "khali values", "empty values"
    ]):
        return df.isnull().sum().rename(
            "Missing Values"
        ).to_frame()

    # Duplicate rows
    if "duplicate" in q or "duplicates" in q:
        duplicate_count = int(df.duplicated().sum())

        return (
            f"Dataset mein **{duplicate_count} duplicate rows** hain."
        )

    # Dataset preview
    if any(x in q for x in [
        "show data", "preview", "first rows", "sample data"
    ]):
        return df.head(10)

    # Statistics
    if any(x in q for x in [
        "describe", "summary statistics", "statistical summary"
    ]):
        return df.describe(include="all").T

    # Most common category
    if any(x in q for x in [
        "most students", "most common", "most frequent",
        "highest count", "maximum students",
        "sabse jyada students", "sabse zyada students"
    ]):
        if "city" in q:
            column = find_column(df, ["city"])
        elif "department" in q:
            column = find_column(df, ["department"])
        elif "gender" in q:
            column = find_column(df, ["gender"])
        elif "course" in q:
            column = find_column(df, ["course"])
        elif "grade" in q:
            column = find_column(df, ["grade"])
        else:
            column = None

        if column is not None:
            counts = df[column].value_counts().rename("Count").to_frame()
            return counts

        return (
            "Please mention a category such as city, department, "
            "gender or course."
        )

    # Grouped average
    if any(x in q for x in [
        "average", "mean", "avg"
    ]):
        if "salary" in q:
            value_column = find_column(df, ["salary"], True)
        elif "marks" in q or "score" in q:
            value_column = find_column(df, ["marks", "score"], True)
        elif "age" in q:
            value_column = find_column(df, ["age"], True)
        elif "attendance" in q:
            value_column = find_column(df, ["attendance"], True)
        elif "fees" in q:
            value_column = find_column(df, ["fees"], True)
        elif "experience" in q:
            value_column = find_column(df, ["experience"], True)
        else:
            value_column = None

        if value_column is None:
            if not numeric_df.empty:
                return numeric_df.mean().rename(
                    "Average"
                ).to_frame()

            return "Dataset mein numerical columns nahi mile."

        if "department" in q:
            group_column = find_column(df, ["department"])
        elif "city" in q:
            group_column = find_column(df, ["city"])
        elif "course" in q:
            group_column = find_column(df, ["course"])
        else:
            group_column = None

        if group_column is not None:
            return (
                df.groupby(group_column)[value_column]
                .mean()
                .sort_values(ascending=False)
                .rename(f"Average {value_column}")
                .to_frame()
            )

        return (
            f"Average {value_column}: "
            f"**{df[value_column].mean():.2f}**"
        )

    # Top N records
    if any(x in q for x in [
        "top", "highest marks", "best students",
        "highest salary", "highest paid"
    ]):
        if "salary" in q:
            column = find_column(df, ["salary"], True)
        elif "age" in q:
            column = find_column(df, ["age"], True)
        elif "attendance" in q:
            column = find_column(df, ["attendance"], True)
        elif "fees" in q:
            column = find_column(df, ["fees"], True)
        else:
            column = find_column(df, ["marks", "score"], True)

        import re

        match = re.search(r"top\s+(\d+)", q)
        n = int(match.group(1)) if match else 5
        n = max(1, min(n, 100))

        if column is not None:
            return df.nlargest(n, column)

        return "Is question ke liye suitable numerical column nahi mila."

    # Highest value
    if any(x in q for x in [
        "highest", "maximum", "max value", "largest"
    ]):
        if "salary" in q:
            column = find_column(df, ["salary"], True)
        elif "marks" in q or "score" in q:
            column = find_column(df, ["marks", "score"], True)
        elif "age" in q:
            column = find_column(df, ["age"], True)
        elif "attendance" in q:
            column = find_column(df, ["attendance"], True)
        elif "fees" in q:
            column = find_column(df, ["fees"], True)
        else:
            column = None

        if column is not None:
            return f"Highest {column}: **{df[column].max()}**"

    # Lowest value
    if any(x in q for x in [
        "lowest", "minimum", "min value", "smallest"
    ]):
        if "salary" in q:
            column = find_column(df, ["salary"], True)
        elif "marks" in q or "score" in q:
            column = find_column(df, ["marks", "score"], True)
        elif "age" in q:
            column = find_column(df, ["age"], True)
        elif "attendance" in q:
            column = find_column(df, ["attendance"], True)
        elif "fees" in q:
            column = find_column(df, ["fees"], True)
        else:
            column = None

        if column is not None:
            return f"Lowest {column}: **{df[column].min()}**"

    return (
        "Abhi main is question ko samajh nahi paaya. "
        "Neeche diye examples try karo:\n\n"
        "- How many rows are there?\n"
        "- Show missing values\n"
        "- Show duplicate rows\n"
        "- What is the average salary?\n"
        "- What is the average marks?\n"
        "- Show top 5 students by marks\n"
        "- Which city has the most students?\n"
        "- Show average salary by department\n"
        "- What are the column names?"
    )


# ----------------------------------
# FILE UPLOAD
# ----------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # ----------------------------------
        # OVERVIEW
        # ----------------------------------

        st.header("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric(
            "Missing Cells",
            int(df.isnull().sum().sum())
        )
        col4.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        # ----------------------------------
        # DATA PREVIEW
        # ----------------------------------

        st.header("📋 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        # ----------------------------------
        # COLUMN INFORMATION
        # ----------------------------------

        st.header("🔎 Column Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info, use_container_width=True)

        # ----------------------------------
        # MISSING VALUES
        # ----------------------------------

        st.header("🕳️ Missing Values Analysis")

        missing = df.isnull().sum()
        missing_table = missing[missing > 0].rename(
            "Missing Values"
        ).to_frame()

        if missing_table.empty:
            st.success("Kisi bhi column mein missing values nahi hain.")
        else:
            st.dataframe(
                missing_table,
                use_container_width=True
            )

        # ----------------------------------
        # DUPLICATE ROWS
        # ----------------------------------

        st.header("🧹 Duplicate Rows")

        duplicate_count = int(df.duplicated().sum())

        if duplicate_count == 0:
            st.success("Koi duplicate row nahi mili.")
        else:
            st.warning(
                f"{duplicate_count} duplicate rows mili hain."
            )

            if st.checkbox("Show duplicate rows"):
                st.dataframe(
                    df[df.duplicated(keep=False)],
                    use_container_width=True
                )

            if st.button("Remove Duplicates from Working Dataset"):
                df = df.drop_duplicates().copy()
                st.success(
                    "Duplicates working dataset se remove ho gayi hain."
                )
                st.info(
                    "Original uploaded file change nahi hui. "
                    "Updated dataset download kar sakte ho."
                )

        # ----------------------------------
        # NUMERICAL STATISTICS
        # ----------------------------------

        numeric_df = df.select_dtypes(include=np.number)

        st.header("📈 Numerical Statistics")

        if numeric_df.empty:
            st.info("Is dataset mein numerical columns nahi mile.")
        else:
            st.dataframe(
                numeric_df.describe().T,
                use_container_width=True
            )

        # ----------------------------------
        # CHARTS
        # ----------------------------------

        st.header("📊 Create Your Chart")

        chart_mode = st.radio(
            "Chart category",
            ["Numerical Chart", "Category Count"],
            horizontal=True
        )

        if chart_mode == "Numerical Chart":

            if not numeric_df.empty:

                selected_column = st.selectbox(
                    "Select numerical column",
                    numeric_df.columns
                )

                chart_type = st.selectbox(
                    "Select chart type",
                    [
                        "Histogram",
                        "Bar Chart",
                        "Line Chart",
                        "Box Plot"
                    ]
                )

                if st.button("Generate Numerical Chart"):

                    values = numeric_df[selected_column].dropna()

                    if values.empty:
                        st.warning("Is column mein valid values nahi hain.")
                    else:
                        fig, ax = plt.subplots()

                        if chart_type == "Histogram":
                            ax.hist(values, bins=10)
                            ax.set_xlabel(selected_column)
                            ax.set_ylabel("Frequency")

                        elif chart_type == "Bar Chart":
                            values.head(30).plot(
                                kind="bar",
                                ax=ax
                            )
                            ax.set_ylabel(selected_column)

                        elif chart_type == "Line Chart":
                            values.reset_index(
                                drop=True
                            ).plot(
                                kind="line",
                                ax=ax
                            )
                            ax.set_ylabel(selected_column)

                        else:
                            ax.boxplot(values)
                            ax.set_ylabel(selected_column)

                        ax.set_title(
                            f"{chart_type}: {selected_column}"
                        )

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            else:
                st.info("Numerical chart ke liye numerical column chahiye.")

        else:

            category_columns = list(
                df.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns
            )

            if category_columns:

                category_column = st.selectbox(
                    "Select category column",
                    category_columns
                )

                if st.button("Generate Category Chart"):

                    counts = (
                        df[category_column]
                        .fillna("Missing")
                        .value_counts()
                        .head(20)
                    )

                    fig, ax = plt.subplots()
                    counts.plot(kind="bar", ax=ax)

                    ax.set_title(
                        f"Count by {category_column}"
                    )
                    ax.set_xlabel(category_column)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(
                        counts.rename("Count").to_frame()
                    )

            else:
                st.info("Category chart ke liye text column nahi mila.")

        # ----------------------------------
        # GROUP COMPARISON
        # ----------------------------------

        st.header("🏙️ Compare Groups")

        all_columns = list(df.columns)

        if len(all_columns) >= 2:

            group_column = st.selectbox(
                "Choose group column",
                all_columns,
                key="group_column"
            )

            if not numeric_df.empty:

                value_column = st.selectbox(
                    "Choose numerical column",
                    numeric_df.columns,
                    key="value_column"
                )

                comparison_type = st.selectbox(
                    "Choose comparison",
                    ["Average", "Total", "Count", "Maximum", "Minimum"]
                )

                if st.button("Compare Groups"):

                    grouped = df.groupby(
                        group_column,
                        dropna=False
                    )[value_column]

                    if comparison_type == "Average":
                        result = grouped.mean()

                    elif comparison_type == "Total":
                        result = grouped.sum()

                    elif comparison_type == "Count":
                        result = grouped.count()

                    elif comparison_type == "Maximum":
                        result = grouped.max()

                    else:
                        result = grouped.min()

                    result = result.sort_values(
                        ascending=False
                    ).rename(comparison_type).to_frame()

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.bar_chart(result)

        # ----------------------------------
        # ASK QUESTIONS
        # ----------------------------------

        st.divider()
        st.header("💬 Ask Your Data Analyst")

        st.write(
            "Dataset ke baare mein English mein question likho."
        )

        question = st.text_input(
            "Your question",
            placeholder="Which city has the most students?"
        )

        if st.button("Analyze My Question"):

            if question.strip():

                with st.spinner("Analyzing your dataset..."):

                    result = analyze_question(
                        df,
                        question
                    )

                st.subheader("✅ Analysis Result")

                if isinstance(result, pd.DataFrame):
                    st.dataframe(
                        result,
                        use_container_width=True
                    )
                else:
                    st.markdown(str(result))

            else:
                st.warning("Please enter a question.")

        # ----------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------

        st.divider()
        st.header("📥 Download Reports")

        report_text = make_report(df)

        st.download_button(
            label="Download Analysis Report (TXT)",
            data=report_text,
            file_name="data_analysis_report.txt",
            mime="text/plain"
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Dataset (CSV)",
            data=csv_data,
            file_name="analyzed_dataset.csv",
            mime="text/csv"
        )

        missing_csv = (
            df.isnull()
            .sum()
            .rename("Missing Values")
            .to_frame()
            .to_csv()
            .encode("utf-8")
        )

        st.download_button(
            label="Download Missing Values Report (CSV)",
            data=missing_csv,
            file_name="missing_values_report.csv",
            mime="text/csv"
        )

        st.success(
            "Your local data analysis tools are ready to use!"
        )

    except Exception as error:
        st.error(f"Something went wrong: {error}")

else:
    st.info("Please upload your CSV or Excel file to begin.")

st.caption(
    "AI Data Analyst Agent V3 | Local Analysis | No Paid API Required"
)