import os
from dotenv import load_dotenv
load_dotenv(override=True)

from nicegui import ui
import pandas as pd
import io
from chatbot import FinancialAnalyzer


# ============================================================
# GLOBAL STATE
# ============================================================

analyzer = None


# ============================================================
# GLOBAL STYLING
# ============================================================

ui.add_head_html("""
<style>

html {
    scroll-behavior: smooth;
}

body {
    background: #0b0f19;
    margin: 0;
    overflow-x: hidden;
}

/* ------------------------------------------------------------
   GENERAL
   ------------------------------------------------------------ */

.q-card {
    border-radius: 18px;
}

.financial-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #ffffff !important;
}

.financial-subtitle {
    font-size: 17px;
    color: #9ca3af !important;
}

/* ------------------------------------------------------------
   STAT CARDS
   ------------------------------------------------------------ */

.stat-card {
    min-width: 210px;
    border-radius: 18px;
    padding: 20px;

    background: rgba(17, 24, 39, 0.85);

    border: 1px solid rgba(255,255,255,0.08);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}

.stat-card:hover {
    transform: translateY(-4px);

    border-color: rgba(255,255,255,0.20);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.25);
}

/* ------------------------------------------------------------
   NAVIGATION BUTTONS
   ------------------------------------------------------------ */

.nav-btn {
    background: linear-gradient(
        135deg,
        #60a5fa,
        #3b82f6
    ) !important;

    color: white !important;

    border: 1px solid rgba(255,255,255,0.18);

    border-radius: 14px;

    box-shadow:
        0 8px 25px rgba(59,130,246,0.20),
        inset 0 1px 1px rgba(255,255,255,0.25);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

.nav-btn:hover {
    transform: translateY(-3px);

    box-shadow:
        0 12px 30px rgba(59,130,246,0.35),
        inset 0 1px 1px rgba(255,255,255,0.35);
}

/* ------------------------------------------------------------
   AI INPUT
   ------------------------------------------------------------ */

.ai-input {
    border-radius: 16px;
}

/* ------------------------------------------------------------
   UPLOAD AREA
   ------------------------------------------------------------ */

.upload-box {
    width: 100%;
    border-radius: 18px;

    background: rgba(17,24,39,0.65);

    border: 1px dashed rgba(96,165,250,0.45);

    padding: 20px;

    transition: all 0.25s ease;
}

.upload-box:hover {
    border-color: rgba(96,165,250,0.8);

    box-shadow:
        0 8px 30px rgba(59,130,246,0.12);
}

/* ------------------------------------------------------------
   CHAT CARDS
   ------------------------------------------------------------ */

.user-message {
    background: rgba(30,64,175,0.45) !important;

    border: 1px solid rgba(96,165,250,0.25);

    border-radius: 16px;
}

.ai-message {
    background: rgba(17,24,39,0.90) !important;

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 16px;
}

/* ------------------------------------------------------------
   DATA TABLE
   ------------------------------------------------------------ */

.data-table {
    width: 100%;
}

/* ------------------------------------------------------------
   RESPONSIVE
   ------------------------------------------------------------ */

@media (max-width: 768px) {

    .financial-title {
        font-size: 32px;
    }

    .financial-subtitle {
        font-size: 15px;
    }

    .stat-card {
        min-width: 100%;
    }

}

</style>
""")


# ============================================================
# PAGE SETTINGS
# ============================================================

ui.page_title("AI Financial Data Analyst")


# ============================================================
# HEADER
# ============================================================

with ui.header().classes(
    "bg-gray-950 text-white items-center px-6"
):

    ui.label(
        "💰 Financial AI"
    ).classes(
        "text-2xl font-bold"
    )

    ui.space()

    ui.label(
        "● AI Online"
    ).classes(
        "text-green-400 font-semibold"
    )


# ============================================================
# LEFT DRAWER
# ============================================================

with ui.left_drawer(
    value=False
).classes(
    "bg-gray-900 text-white p-5"
):

    ui.label(
        "💰 Financial AI"
    ).classes(
        "text-2xl font-bold mb-2"
    )

    ui.label(
        "Your AI-powered financial data analyst."
    ).classes(
        "text-gray-400 mb-6"
    )

    ui.separator()

    ui.label(
        "📌 What you can ask"
    ).classes(
        "text-lg font-semibold mt-5 mb-4"
    )

    questions = [
        "Total / average revenue",
        "Net income & profit",
        "Company comparisons",
        "Highest / lowest values",
        "Financial trends",
        "Year-over-year changes",
        "Profit margins",
        "Debt & liabilities",
        "Dataset statistics",
        "Missing values",
    ]

    for question in questions:

        ui.label(
            "• " + question
        ).classes(
            "text-gray-300 text-sm mb-3"
        )


# ============================================================
# MAIN CONTENT
# ============================================================

with ui.column().classes(
    "w-full max-w-7xl mx-auto p-6 md:p-8"
).props(
    "id=dashboard-section"
):

    # ========================================================
    # PAGE TITLE
    # ========================================================

    ui.label(
        "💰 AI Financial Data Analyst"
    ).classes(
        "financial-title"
    )

    ui.label(
        "Upload a financial CSV and explore it using natural language."
    ).classes(
        "financial-subtitle mb-8"
    )


    # ========================================================
    # NAVIGATION
    # ========================================================

    with ui.row().classes(
        "w-full items-center gap-3 mb-8 flex-wrap"
    ):

        # ----------------------------------------------------
        # DASHBOARD
        # ----------------------------------------------------

        ui.button(
            "📊 Dashboard",
            on_click=lambda: ui.run_javascript(
                """
                document.getElementById(
                    'dashboard-section'
                ).scrollIntoView({
                    behavior: 'smooth'
                });
                """
            )
        ).props(
            "unelevated"
        ).classes(
            "nav-btn px-6 py-3 rounded-xl text-white font-semibold"
        )


        # ----------------------------------------------------
        # AI ANALYST
        # ----------------------------------------------------

        ui.button(
            "💬 AI Analyst",
            on_click=lambda: ui.run_javascript(
                """
                document.getElementById(
                    'ai-section'
                ).scrollIntoView({
                    behavior: 'smooth'
                });
                """
            )
        ).props(
            "unelevated"
        ).classes(
            "nav-btn px-6 py-3 rounded-xl text-white font-semibold"
        )


        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        ui.button(
            "📋 Dataset",
            on_click=lambda: ui.run_javascript(
                """
                document.getElementById(
                    'dataset-section'
                ).scrollIntoView({
                    behavior: 'smooth'
                });
                """
            )
        ).props(
            "unelevated"
        ).classes(
            "nav-btn px-6 py-3 rounded-xl text-white font-semibold"
        )


    # ========================================================
    # UPLOAD SECTION
    # ========================================================

    ui.label(
        "📤 Upload Financial Data"
    ).classes(
        "text-2xl font-bold text-white"
    )

    ui.label(
        "Choose a CSV file containing your financial data."
    ).classes(
        "text-gray-400 mb-4"
    )


    upload_status = ui.label(
        "No file uploaded yet."
    ).classes(
        "text-gray-400"
    )


    dataset_info = ui.label(
        ""
    ).classes(
        "text-gray-300 mt-2"
    )


    # ========================================================
    # DASHBOARD CONTAINERS
    # ========================================================

    stats_container = ui.row().classes(
        "w-full gap-4 mt-6 flex-wrap"
    )


    preview_container = ui.column().classes(
        "w-full mt-6"
    ).props(
        "id=dataset-section"
    )


    # ========================================================
    # UPLOAD FUNCTION
    # ========================================================

    async def handle_upload(event):

        global analyzer

        try:

            # ------------------------------------------------
            # READ FILE
            # ------------------------------------------------

            contents = await event.file.read()

            df = pd.read_csv(
                io.BytesIO(contents)
            )


            # ------------------------------------------------
            # CHECK EMPTY DATASET
            # ------------------------------------------------

            if df.empty:

                upload_status.set_text(
                    "❌ The uploaded CSV is empty."
                )

                return


            if len(df.columns) == 0:

                upload_status.set_text(
                    "❌ The CSV does not contain any columns."
                )

                return


            # ------------------------------------------------
            # CREATE AI ANALYZER
            # ------------------------------------------------

            analyzer = FinancialAnalyzer(df)


            # ------------------------------------------------
            # SUCCESS MESSAGE
            # ------------------------------------------------

            upload_status.set_text(
                f"✅ Successfully uploaded: {event.file.name}"
            )


            dataset_info.set_text(
                f"Rows: {df.shape[0]:,}  |  "
                f"Columns: {df.shape[1]:,}  |  "
                f"🧠 AI Analyzer Ready"
            )


            # ------------------------------------------------
            # DATA QUALITY
            # ------------------------------------------------

            total_cells = (
                df.shape[0] *
                df.shape[1]
            )


            missing_values = int(
                df.isna().sum().sum()
            )


            duplicate_rows = int(
                df.duplicated().sum()
            )


            if total_cells > 0:

                completeness = (
                    (
                        1 -
                        missing_values /
                        total_cells
                    ) * 100
                )

            else:

                completeness = 100


            # ------------------------------------------------
            # CLEAR PREVIOUS DATA
            # ------------------------------------------------

            stats_container.clear()

            preview_container.clear()


            # =================================================
            # STAT CARDS
            # =================================================

            with stats_container:

                # ------------------------------------------------
                # ROWS
                # ------------------------------------------------

                with ui.card().classes(
                    "stat-card text-white"
                ):

                    ui.label(
                        "📊 Rows"
                    ).classes(
                        "text-gray-400"
                    )

                    ui.label(
                        f"{len(df):,}"
                    ).classes(
                        "text-3xl font-bold"
                    )


                # ------------------------------------------------
                # COLUMNS
                # ------------------------------------------------

                with ui.card().classes(
                    "stat-card text-white"
                ):

                    ui.label(
                        "📋 Columns"
                    ).classes(
                        "text-gray-400"
                    )

                    ui.label(
                        f"{len(df.columns):,}"
                    ).classes(
                        "text-3xl font-bold"
                    )


                # ------------------------------------------------
                # COMPLETENESS
                # ------------------------------------------------

                with ui.card().classes(
                    "stat-card text-white"
                ):

                    ui.label(
                        "🧹 Completeness"
                    ).classes(
                        "text-gray-400"
                    )

                    ui.label(
                        f"{completeness:.1f}%"
                    ).classes(
                        "text-3xl font-bold"
                    )


                # ------------------------------------------------
                # DUPLICATES
                # ------------------------------------------------

                with ui.card().classes(
                    "stat-card text-white"
                ):

                    ui.label(
                        "🔁 Duplicate Rows"
                    ).classes(
                        "text-gray-400"
                    )

                    ui.label(
                        f"{duplicate_rows:,}"
                    ).classes(
                        "text-3xl font-bold"
                    )


            # =================================================
            # DATA PREVIEW
            # =================================================

            with preview_container:

                ui.label(
                    "📋 Data Preview"
                ).classes(
                    "text-2xl font-bold text-white"
                )


                ui.label(
                    f"Showing your complete dataset — "
                    f"{len(df):,} rows available"
                ).classes(
                    "text-gray-400 mb-3"
                )


                # ------------------------------------------------
                # TABLE COLUMNS
                # ------------------------------------------------

                columns = [
                    {
                        "name": str(column),
                        "label": str(column),
                        "field": str(column),
                        "sortable": True
                    }

                    for column in df.columns
                ]


                # ------------------------------------------------
                # TABLE ROWS
                # ------------------------------------------------

                rows = (
                    df.head(10)
                    .fillna("")
                    .to_dict(
                        orient="records"
                    )
                )


                # ------------------------------------------------
                # TABLE
                # ------------------------------------------------

                table = ui.table(
                    columns=columns,
                    rows=rows,
                    row_key=columns[0]["name"],
                    pagination={
                        "rowsPerPage": 10
                    }
                ).classes(
                    "w-full"
                )


                table.props(
                    "flat bordered"
                )


        except Exception as e:

            upload_status.set_text(
                f"❌ Unable to read CSV: {e}"
            )


    # ========================================================
    # UPLOAD BUTTON
    # ========================================================

    ui.upload(
        label="Upload CSV",
        on_upload=handle_upload,
        auto_upload=True
    ).props(
        "accept=.csv"
    ).classes(
        "w-full"
    )


    # ========================================================
    # AI CHAT SECTION
    # ========================================================

    ui.separator().classes(
        "my-8"
    )


    ui.element(
        "div"
    ).props(
        "id=ai-section"
    )


    ui.label(
        "💬 AI ChatBot"
    ).classes(
        "text-3xl font-bold text-white"
    )


    ui.label(
        "Ask questions about your uploaded financial dataset."
    ).classes(
        "text-gray-400 mb-4"
    )


    # ========================================================
    # CHAT MESSAGES
    # ========================================================

    chat_container = ui.column().classes(
        "w-full gap-3"
    )


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    question_input = ui.input(
        placeholder=(
            "Ask something like: "
            "What are the top 5 companies by market cap?"
        )
    ).classes(
        "w-full ai-input"
    ).props(
        "outlined dark"
    )


    # ========================================================
    # ASK AI FUNCTION
    # ========================================================

    async def ask_ai():

        question = (
            question_input.value or ""
        ).strip()


        # ----------------------------------------------------
        # EMPTY QUESTION
        # ----------------------------------------------------

        if not question:

            ui.notify(
                "Please enter a question.",
                type="warning"
            )

            return


        # ----------------------------------------------------
        # NO DATASET
        # ----------------------------------------------------

        if analyzer is None:

            ui.notify(
                "Please upload a CSV first.",
                type="warning"
            )

            return


        # ----------------------------------------------------
        # SHOW USER QUESTION
        # ----------------------------------------------------

        with chat_container:

            with ui.card().classes(
                "user-message w-full text-white"
            ):

                ui.label(
                    "👤 You"
                ).classes(
                    "font-bold"
                )

                ui.label(
                    question
                )


        question_input.value = ""


        # ----------------------------------------------------
        # THINKING MESSAGE
        # ----------------------------------------------------

        with chat_container:

            thinking = ui.card().classes(
                "ai-message w-full text-white"
            )

            with thinking:

                ui.label(
                    "🤖 Financial AI"
                ).classes(
                    "font-bold"
                )

                ui.label(
                    "Analyzing your financial data..."
                ).classes(
                    "text-gray-400"
                )


        # ----------------------------------------------------
        # ASK ANALYZER
        # ----------------------------------------------------

        try:

            result = analyzer.answer_question(
                question
            )


            print(
                "\n-------- AI RESULT --------"
            )

            print(result)


            print(
                "\n-------- CHART --------"
            )

            print(
                result.get("chart")
            )


            print(
                "---------------------------\n"
            )


            # ------------------------------------------------
            # REMOVE THINKING MESSAGE
            # ------------------------------------------------

            thinking.delete()


            # ------------------------------------------------
            # AI RESPONSE
            # ------------------------------------------------

            with chat_container:

                with ui.card().classes(
                    "ai-message w-full text-white"
                ):

                    ui.label(
                        "🤖 Financial AI"
                    ).classes(
                        "font-bold"
                    )


                    # ------------------------------------------------
                    # ANSWER
                    # ------------------------------------------------

                    answer = result.get(
                        "answer",
                        "I couldn't generate an answer."
                    )


                    ui.markdown(
                        str(answer)
                    )


                    # =================================================
                    # VISUALIZATION
                    # =================================================

                    chart = result.get(
                        "chart"
                    )


                    if chart:

                        ui.label(
                            "📊 Visualization"
                        ).classes(
                            "text-lg font-semibold mt-4"
                        )


                        try:

                            import plotly.graph_objects as go


                            # ------------------------------------------------
                            # GET CHART INFORMATION
                            # ------------------------------------------------

                            chart_type = chart.get(
                                "type",
                                "bar"
                            )


                            chart_df = chart.get(
                                "data"
                            )


                            x_column = chart.get(
                                "x"
                            )


                            y_column = chart.get(
                                "y"
                            )


                            title = chart.get(
                                "title",
                                "Financial Analysis"
                            )


                            # ------------------------------------------------
                            # CHECK DATA
                            # ------------------------------------------------

                            if chart_df is None:

                                raise ValueError(
                                    "No chart data was returned "
                                    "by the AI analyzer."
                                )


                            if chart_df.empty:

                                raise ValueError(
                                    "The chart data is empty."
                                )


                            if x_column not in chart_df.columns:

                                raise ValueError(
                                    f"Column '{x_column}' "
                                    "not found in chart data."
                                )


                            if y_column not in chart_df.columns:

                                raise ValueError(
                                    f"Column '{y_column}' "
                                    "not found in chart data."
                                )


                            # ------------------------------------------------
                            # CREATE FIGURE
                            # ------------------------------------------------

                            fig = go.Figure()


                            # ------------------------------------------------
                            # BAR
                            # ------------------------------------------------

                            if chart_type == "bar":

                                fig.add_trace(
                                    go.Bar(
                                        x=chart_df[
                                            x_column
                                        ].tolist(),

                                        y=chart_df[
                                            y_column
                                        ].tolist(),

                                        name=y_column
                                    )
                                )


                            # ------------------------------------------------
                            # LINE
                            # ------------------------------------------------

                            elif chart_type == "line":

                                fig.add_trace(
                                    go.Scatter(
                                        x=chart_df[
                                            x_column
                                        ].tolist(),

                                        y=chart_df[
                                            y_column
                                        ].tolist(),

                                        mode="lines+markers",

                                        name=y_column
                                    )
                                )


                            # ------------------------------------------------
                            # SCATTER
                            # ------------------------------------------------

                            elif chart_type == "scatter":

                                fig.add_trace(
                                    go.Scatter(
                                        x=chart_df[
                                            x_column
                                        ].tolist(),

                                        y=chart_df[
                                            y_column
                                        ].tolist(),

                                        mode="markers",

                                        name=y_column
                                    )
                                )


                            # ------------------------------------------------
                            # DEFAULT
                            # ------------------------------------------------

                            else:

                                fig.add_trace(
                                    go.Bar(
                                        x=chart_df[
                                            x_column
                                        ].tolist(),

                                        y=chart_df[
                                            y_column
                                        ].tolist(),

                                        name=y_column
                                    )
                                )


                            # ------------------------------------------------
                            # CHART DESIGN
                            # ------------------------------------------------

                            fig.update_layout(

                                title=title,

                                template="plotly_dark",

                                height=500,

                                xaxis_title=x_column,

                                yaxis_title=y_column,

                                margin=dict(
                                    l=50,
                                    r=50,
                                    t=80,
                                    b=100
                                )
                            )


                            # ------------------------------------------------
                            # DISPLAY CHART
                            # ------------------------------------------------

                            ui.plotly(
                                fig
                            ).classes(
                                "w-full"
                            )


                        except Exception as chart_error:

                            ui.label(
                                "⚠️ Chart could not be displayed: "
                                f"{chart_error}"
                            ).classes(
                                "text-orange-400"
                            )


        # ----------------------------------------------------
        # MAIN ERROR
        # ----------------------------------------------------

        except Exception as e:

            thinking.delete()


            with chat_container:

                with ui.card().classes(
                    "w-full bg-red-950 text-white"
                ):

                    ui.label(
                        "❌ Something went wrong"
                    ).classes(
                        "font-bold"
                    )

                    ui.label(
                        str(e)
                    )


    # ========================================================
    # ASK AI BUTTON
    # ========================================================

    ui.button(
        "🚀 Ask AI",
        on_click=ask_ai
    ).classes(
        "nav-btn mt-3 px-8 py-3 text-lg font-semibold rounded-xl"
    ).props(
        "unelevated"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

port = int(os.environ.get("PORT", 8080))

ui.run(
    host="0.0.0.0",
    port=port,
    reload=False,
    reconnect_timeout=30,
    title="AI Financial Data Analyst"
)