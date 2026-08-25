import streamlit as st
import pandas as pd
import plotly.express as px
from chatbot import FinancialAnalyzer
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Financial Data Analyst",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f9fafb;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .card {
        background: linear-gradient(
            145deg,
            #111827,
            #172033
        );
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.20);
    }

    .metric-title {
        color: #9ca3af;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #f9fafb;
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #f9fafb;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .upload-box {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .welcome-box {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }

    .welcome-title {
        font-size: 25px;
        font-weight: 700;
        color: #f9fafb;
    }

    .welcome-text {
        color: #9ca3af;
        font-size: 16px;
        line-height: 1.7;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="main-title" style="font-size:25px;">💰 Financial AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Your AI-powered financial data analyst."
    )

    st.divider()

    st.markdown("### 📌 What you can ask")

    st.markdown("""
    - Total / average revenue
    - Net income & profit
    - Company comparisons
    - Highest / lowest values
    - Financial trends
    - Year-over-year changes
    - Profit margins
    - Debt & liabilities
    - Dataset statistics
    - Missing values
    - Financial summaries
    """)

    st.divider()

    st.markdown("### 💡 Example questions")

    st.caption(
        "Which company has the highest revenue?"
    )

    st.caption(
        "Compare revenue and net income."
    )

    st.caption(
        "What changed between the earliest and latest year?"
    )

    st.caption(
        "Give me a financial summary."
    )

    st.caption(
        "Are there any missing values?"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💰 AI Financial Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a financial CSV and ask questions about your data '
    'using natural language.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Financial Data</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your financial CSV file",
    type=["csv"],
    help="Upload any CSV containing financial or business data."
)


# ============================================================
# NO FILE UPLOADED
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="welcome-box">

            <div class="welcome-title">
                👋 Welcome to Financial AI
            </div>

            <div class="welcome-text">
                Upload a CSV file to start analyzing your
                financial data.<br><br>

                Once uploaded, you can ask questions about
                revenue, profit, companies, years, trends,
                comparisons and much more.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# LOAD CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"❌ Unable to read the CSV file: {e}"
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if df.empty:

    st.error(
        "❌ The uploaded CSV is empty."
    )

    st.stop()


# ============================================================
# CREATE ANALYZER
# ============================================================

analyzer = FinancialAnalyzer(df)

dataset_signature = (
    uploaded_file.name,
    uploaded_file.size
)

if (
    "dataset_signature"
    not in st.session_state
    or
    st.session_state.dataset_signature
    != dataset_signature
):

    st.session_state.messages = []

    st.session_state.dataset_signature = (
        dataset_signature
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

# ============================================================
# DATA QUALITY
# ============================================================

st.markdown(
    '<div class="section-title">🧹 Data Quality</div>',
    unsafe_allow_html=True
)

quality_col1, quality_col2, quality_col3 = st.columns(3)

total_cells = df.shape[0] * df.shape[1]

missing_cells = int(
    df.isna().sum().sum()
)

if total_cells > 0:
    completeness = (
        (1 - missing_cells / total_cells) * 100
    )
else:
    completeness = 100


with quality_col1:

    st.metric(
        "Data Completeness",
        f"{completeness:.1f}%"
    )


with quality_col2:

    duplicate_rows = int(
        df.duplicated().sum()
    )

    st.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )


with quality_col3:

    memory_usage = (
        df.memory_usage(deep=True)
        .sum() / (1024 ** 2)
    )

    st.metric(
        "Memory Usage",
        f"{memory_usage:.2f} MB"
    )
# ============================================================
# STATISTICAL SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📈 Statistical Summary</div>',
    unsafe_allow_html=True
)

statistics = analyzer.get_statistics()

if statistics:

    stats_df = pd.DataFrame(
        statistics
    ).T

    stats_df.index.name = "Column"

    st.dataframe(
        stats_df,
        use_container_width=True
    )

else:

    st.info(
        "No numeric columns were detected in the dataset."
    )

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)

profile = analyzer.get_profile()

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Rows</div>
            <div class="metric-value">
                {profile["rows"]:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Columns</div>
            <div class="metric-value">
                {profile["columns"]:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Missing Values</div>
            <div class="metric-value">
                {profile["missing_values"]:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    numeric_count = len(
        profile["numeric_columns"]
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Numeric Fields</div>
            <div class="metric-value">
                {numeric_count:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📋 Data Preview</div>',
    unsafe_allow_html=True
)

with st.expander(
    "View uploaded dataset",
    expanded=True
):

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=350
    )


# ============================================================
# DETECTED COLUMNS
# ============================================================

with st.expander("🔍 Dataset Information"):

    st.write(
        "**Columns:**"
    )

    st.write(
        profile["column_names"]
    )

    st.write(
        "**Numeric columns:**"
    )

    st.write(
        profile["numeric_columns"]
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# CHAT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask Your Financial Data</div>',
    unsafe_allow_html=True
)


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your uploaded financial data..."
)


if question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your financial data..."
        ):

            result = analyzer.answer_question(
                question
            )

        # Get text answer
        response = result["answer"]

        st.markdown(response)

        # Get chart
        chart = result.get("chart")

        if chart:

            chart_df = chart["data"]

            if chart["type"] == "bar":

                fig = px.bar(
                    chart_df,
                    x=chart["x"],
                    y=chart["y"],
                    title=chart["title"]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            elif chart["type"] == "line":

                fig = px.line(
                    chart_df,
                    x=chart["x"],
                    y=chart["y"],
                    title=chart["title"]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.messages:

    if st.button(
        "🗑️ Clear Conversation"
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        font-size:13px;
        padding:10px;
    ">
        AI Financial Data Analyst • Pandas + Groq + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)