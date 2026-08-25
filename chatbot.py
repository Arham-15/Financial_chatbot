import os
import json
import re
import pandas as pd
import streamlit as st
from groq import Groq


class FinancialAnalyzer:

    def __init__(self, df):

        self.df = df.copy()

        # =====================================================
        # CLEAN COLUMN NAMES
        # =====================================================

        self.df.columns = self.df.columns.astype(str).str.strip()

        # =====================================================
        # CONVERT NUMERIC-LOOKING COLUMNS
        # =====================================================

        for column in self.df.columns:

            converted = pd.to_numeric(self.df[column], errors="coerce")

            if len(self.df) > 0:

                conversion_ratio = converted.notna().sum() / len(self.df)

                if conversion_ratio >= 0.5:
                    self.df[column] = converted

        # =====================================================
        # COLUMN LOOKUP
        # =====================================================

        self.column_lookup = {
            self.normalize_text(column): column for column in self.df.columns
        }

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_text(text):

        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    # =========================================================
    # DATASET PROFILE
    # =========================================================

    def get_profile(self):

        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()

        categorical_columns = self.df.select_dtypes(exclude="number").columns.tolist()

        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": int(self.df.isna().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

    # =========================================================
    # FINANCIAL STATISTICS
    # =========================================================

    def get_statistics(self):

        statistics = {}

        numeric_columns = self.df.select_dtypes(include="number").columns

        for column in numeric_columns:

            series = self.df[column].dropna()

            if series.empty:
                continue

            statistics[column] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std()),
            }

        return statistics

    # =========================================================
    # DATASET SUMMARY
    # =========================================================

    def get_summary(self):

        summary = {}

        for column in self.df.select_dtypes(include="number").columns:

            series = self.df[column].dropna()

            if len(series) == 0:
                continue

            summary[column] = {
                "sum": float(series.sum()),
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

        return summary

    # =========================================================
    # SAMPLE DATA
    # =========================================================

    def get_sample(self, rows=10):

        sample = self.df.head(rows).copy()

        sample = sample.fillna("")

        return sample.to_dict(orient="records")

    # =========================================================
    # GET CATEGORICAL VALUES
    # =========================================================

    def get_categorical_values(self, max_values=300):

        values = {}

        for column in self.df.select_dtypes(exclude="number").columns:

            unique_values = (
                self.df[column].dropna().astype(str).str.strip().unique().tolist()
            )

            if len(unique_values) <= max_values:

                values[column] = unique_values

        return values

    # =========================================================
    # GROQ CLIENT
    # =========================================================

    def get_client(self):

        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY was not found.")

        return Groq(api_key=api_key)

    # =========================================================
    # FIND COLUMN
    # =========================================================

    def resolve_column(self, column_name):

        if not column_name:
            return None

        normalized = self.normalize_text(column_name)

        # Exact match
        if normalized in self.column_lookup:

            return self.column_lookup[normalized]

        # Partial match
        for key, actual_column in self.column_lookup.items():

            if normalized in key or key in normalized:

                return actual_column

        return None

    # =========================================================
    # INFER COLUMNS FROM USER QUESTION
    # =========================================================

    def infer_columns_from_question(self, question):

        question_normalized = self.normalize_text(question)

        matches = []

        # Common financial aliases. The actual dataset column
        # name is always returned; aliases only help recognition.
        aliases = {
            "revenue": ["revenue", "revenues", "sales", "net sales", "total revenue"],
            "net profit": [
                "net profit",
                "net income",
                "profit after tax",
                "pat",
                "net earnings",
            ],
            "gross profit": ["gross profit"],
            "operating profit": ["operating profit", "operating income"],
            "expenses": ["expenses", "expense", "total expenses", "operating expenses"],
            "assets": ["assets", "total assets"],
            "liabilities": ["liabilities", "total liabilities"],
            "equity": ["equity", "shareholders equity", "stockholders equity"],
            "cash flow": ["cash flow", "cash flows"],
            "ebitda": ["ebitda"],
        }

        normalized_columns = {
            column: self.normalize_text(column)
            for column in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[column])
        }

        # First: exact normalized column phrase.
        for column, normalized_column in normalized_columns.items():

            if normalized_column and normalized_column in question_normalized:
                matches.append(column)

        # Second: common financial aliases.
        for canonical, alias_list in aliases.items():

            alias_found = any(
                self.normalize_text(alias) in question_normalized
                for alias in alias_list
            )

            if not alias_found:
                continue

            candidate_columns = []

            for column, normalized_column in normalized_columns.items():

                if canonical == "revenue":
                    if "revenue" in normalized_column or "sales" in normalized_column:
                        candidate_columns.append(column)

                elif canonical == "net profit":
                    if (
                        "net profit" in normalized_column
                        or "net income" in normalized_column
                        or "profit after tax" in normalized_column
                    ):
                        candidate_columns.append(column)

                elif canonical == "gross profit":
                    if "gross profit" in normalized_column:
                        candidate_columns.append(column)

                elif canonical == "operating profit":
                    if (
                        "operating profit" in normalized_column
                        or "operating income" in normalized_column
                    ):
                        candidate_columns.append(column)

                elif canonical == "expenses":
                    if "expense" in normalized_column:
                        candidate_columns.append(column)

                elif canonical == "assets":
                    if "asset" in normalized_column:
                        candidate_columns.append(column)

                elif canonical == "liabilities":
                    if "liabilit" in normalized_column:
                        candidate_columns.append(column)

                elif canonical == "equity":
                    if "equity" in normalized_column:
                        candidate_columns.append(column)

                elif canonical == "cash flow":
                    if "cash flow" in normalized_column:
                        candidate_columns.append(column)

                elif canonical == "ebitda":
                    if "ebitda" in normalized_column:
                        candidate_columns.append(column)

            matches.extend(candidate_columns)

        return list(dict.fromkeys(matches))

    # =========================================================
    # VALIDATE ENTITY / FILTER
    # =========================================================

    def validate_filter(self, filter_data):

        if not filter_data:
            return None

        column_name = filter_data.get("column")

        value = filter_data.get("value")

        if not column_name or value is None:
            return None

        actual_column = self.resolve_column(column_name)

        if actual_column is None:

            return {
                "valid": False,
                "message": (
                    f"The dataset does not contain " f"a column called '{column_name}'."
                ),
            }

        series = self.df[actual_column].dropna().astype(str).str.strip()

        requested_value = str(value).strip()

        # Exact match
        matches = series[series.str.lower() == requested_value.lower()]

        if len(matches) == 0:

            # Partial match
            partial_matches = series[
                series.str.lower().str.contains(requested_value.lower(), regex=False)
            ]

            if len(partial_matches) == 0:

                return {
                    "valid": False,
                    "message": (
                        f"'{value}' is not present in " f"the uploaded dataset."
                    ),
                }

            actual_value = partial_matches.iloc[0]

        else:

            actual_value = matches.iloc[0]

        return {"valid": True, "column": actual_column, "value": actual_value}

        # =========================================================
    # UNDERSTAND QUESTION
    # =========================================================

    def understand_question(self, question):

        client = self.get_client()

        profile = self.get_profile()

        categorical_values = self.get_categorical_values()

        prompt = (
            """
You are a query planner for a financial CSV analysis system.

Your ONLY job is to convert the user's question into a JSON query plan.

You MUST NOT answer the user's question.

AVAILABLE COLUMNS:
"""
            + json.dumps(profile["column_names"], indent=2)
            + """

NUMERIC COLUMNS:
"""
            + json.dumps(profile["numeric_columns"], indent=2)
            + """

CATEGORICAL VALUES:
"""
            + json.dumps(categorical_values, indent=2, default=str)
            + """

ALLOWED OPERATIONS:

overview
missing_values
average
sum
maximum
minimum
ranking
comparison
trend
count

RETURN ONLY VALID JSON.

The JSON must contain these fields:

operation
column
columns
group_by
filter
limit

RULES:

1. Only use columns that actually exist in AVAILABLE COLUMNS.

2. Never invent column names.

3. If the user asks for one numeric metric:
   - put the metric in "column"
   - also put it inside "columns"

4. If the user asks to compare multiple metrics:
   - use operation "comparison"
   - put every requested metric inside "columns"

5. If the user asks to compare one metric across companies/entities:
   - put the metric in "column"
   - put the company/entity column in "group_by"

6. "highest" usually means "maximum" or "ranking".

7. "lowest" usually means "minimum" or "ranking".

8. "average" means "average".

9. "total" means "sum".

10. "top N" means "ranking" and limit should be N.

11. "trend" means "trend".

12. Questions about the dataset itself mean "overview".

13. Questions about missing data mean "missing_values".

14. Questions asking how many records/entities exist mean "count".

15. If the user specifies a company/entity, use the actual categorical column
    from the dataset as the filter column.

16. If the requested entity does not exist, still return it as the filter value.
    The Python application will validate whether it exists.

17. Do not add explanations.

18. Return ONLY the JSON object.

USER QUESTION:
"""
            + question
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # Remove markdown code fences if Groq returns them
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        plan = json.loads(content)

        # -----------------------------------------------------
        # DETERMINISTIC QUESTION ANALYSIS
        # -----------------------------------------------------

        question_normalized = self.normalize_text(question)

        # Detect actual columns mentioned in the question
        detected_columns = []

        for actual_column in self.df.columns:

            normalized_column = self.normalize_text(actual_column)

            if normalized_column and normalized_column in question_normalized:
                detected_columns.append(actual_column)

        # Also use financial aliases
        inferred_columns = self.infer_columns_from_question(question)

        for column in inferred_columns:

            if column not in detected_columns:
                detected_columns.append(column)

        # -----------------------------------------------------
        # DETECT ENTITY QUESTIONS
        # -----------------------------------------------------

        entity_phrases = [
            "which company",
            "which companies",
            "which ticker",
            "which stock",
            "which entity",
            "what company",
            "what stock",
            "who has",
            "who had",
        ]

        asks_for_entity = any(
            phrase in question_normalized
            for phrase in entity_phrases
        )

        # -----------------------------------------------------
        # DETECT HIGHEST / LOWEST
        # -----------------------------------------------------

        highest_phrases = [
            "highest",
            "maximum",
            "max",
            "largest",
            "greatest",
            "top",
        ]

        lowest_phrases = [
            "lowest",
            "minimum",
            "min",
            "smallest",
            "least",
        ]

        asks_highest = any(
            phrase in question_normalized
            for phrase in highest_phrases
        )

        asks_lowest = any(
            phrase in question_normalized
            for phrase in lowest_phrases
        )

        # -----------------------------------------------------
        # FORCE RANKING FOR ENTITY QUESTIONS
        # -----------------------------------------------------

                # -----------------------------------------------------
        # DETECT TOP N / BOTTOM N
        # -----------------------------------------------------

        number_match = re.search(
            r"\b(?:top|bottom|highest|lowest|largest|smallest)\s+(\d+)\b",
            question_normalized
        )

        # Also support:
        # "5 highest companies"
        # "10 largest stocks"
        reverse_number_match = re.search(
            r"\b(\d+)\s+(?:highest|lowest|largest|smallest|top|bottom)\b",
            question_normalized
        )

        if number_match:

            requested_limit = int(number_match.group(1))

        elif reverse_number_match:

            requested_limit = int(reverse_number_match.group(1))

        else:

            requested_limit = 1

        # Never allow an invalid/huge limit
        requested_limit = max(1, min(requested_limit, len(self.df)))

        # -----------------------------------------------------
        # FORCE RANKING FOR RANKING QUESTIONS
        # -----------------------------------------------------

        is_ranking_question = (
            asks_for_entity
            and (asks_highest or asks_lowest)
        ) or (
            "top" in question_normalized
            or "bottom" in question_normalized
            or "largest" in question_normalized
            or "smallest" in question_normalized
        )

        if is_ranking_question and detected_columns:

            plan["operation"] = "ranking"

            plan["column"] = detected_columns[0]

            plan["columns"] = [detected_columns[0]]

            plan["limit"] = requested_limit

            if asks_lowest or "bottom" in question_normalized or "smallest" in question_normalized:

                plan["sort_order"] = "ascending"

            else:

                plan["sort_order"] = "descending"

        # -----------------------------------------------------
        # NORMAL COLUMN FALLBACK
        # -----------------------------------------------------

        if not plan.get("column") and detected_columns:
            plan["column"] = detected_columns[0]

        if not plan.get("columns") and detected_columns:
            plan["columns"] = detected_columns

        if (
            plan.get("operation") == "comparison"
            and detected_columns
        ):
            plan["columns"] = detected_columns

        return plan

        # -----------------------------------------------------
        # Deterministic column detection
        # -----------------------------------------------------

        inferred_columns = self.infer_columns_from_question(question)

        if not plan.get("column") and inferred_columns:
            plan["column"] = inferred_columns[0]

        if not plan.get("columns") and inferred_columns:
            plan["columns"] = inferred_columns

        if plan.get("operation") == "comparison" and inferred_columns:
            plan["columns"] = inferred_columns

        return plan
            # -----------------------------------------------------
        # Deterministic entity-ranking detection
        # -----------------------------------------------------

        question_normalized = self.normalize_text(question)

        entity_words = [
            "which company",
            "which companies",
            "which stock",
            "which stocks",
            "which ticker",
            "which entity",
            "who has",
            "what company",
        ]

        highest_words = [
            "highest",
            "maximum",
            "max",
            "largest",
            "greatest",
            "top",
        ]

        lowest_words = [
            "lowest",
            "minimum",
            "min",
            "smallest",
            "least",
        ]

        asks_for_entity = any(
            word in question_normalized
            for word in entity_words
        )

        asks_highest = any(
            word in question_normalized
            for word in highest_words
        )

        asks_lowest = any(
            word in question_normalized
            for word in lowest_words
        )

        if asks_for_entity and (asks_highest or asks_lowest):

            if inferred_columns:
                plan["operation"] = "ranking"
                plan["column"] = inferred_columns[0]
                plan["columns"] = [inferred_columns[0]]

                if asks_lowest:
                    plan["limit"] = 1
                    plan["sort_order"] = "ascending"
                else:
                    plan["limit"] = 1
                    plan["sort_order"] = "descending"

    # =========================================================
    # EXECUTE QUESTION WITH PANDAS
    # =========================================================

    def execute_query(self, plan):

        operation = plan.get("operation")

        column_name = plan.get("column")

        group_by = plan.get("group_by")

        filter_data = plan.get("filter")

        limit = plan.get("limit", 10)

        # =====================================================
        # VALIDATE COLUMN
        # =====================================================

        column = None

        if column_name:

            column = self.resolve_column(column_name)

            if column is None:

                return {
                    "success": False,
                    "message": (
                        f"The column '{column_name}' " "does not exist in the dataset."
                    ),
                }

        # =====================================================
        # VALIDATE GROUP COLUMN
        # =====================================================

        group_column = None

        if group_by:

            group_column = self.resolve_column(group_by)

            if group_column is None:

                return {
                    "success": False,
                    "message": (
                        f"The grouping column " f"'{group_by}' does not exist."
                    ),
                }

        # =====================================================
        # VALIDATE FILTER
        # =====================================================

        validated_filter = self.validate_filter(filter_data)

        if validated_filter:

            if not validated_filter["valid"]:

                return {"success": False, "message": (validated_filter["message"])}

            filter_column = validated_filter["column"]

            filter_value = validated_filter["value"]

            working_df = self.df[
                self.df[filter_column].astype(str).str.strip().str.lower()
                == str(filter_value).strip().lower()
            ].copy()

        else:

            working_df = self.df.copy()

        # =====================================================
        # EMPTY DATASET AFTER FILTER
        # =====================================================

        if working_df.empty:

            return {
                "success": False,
                "message": (
                    "No matching records were found " "in the uploaded dataset."
                ),
            }

        # =====================================================
        # OVERVIEW
        # =====================================================

        if operation == "overview":

            return {
                "success": True,
                "operation": "overview",
                "result": self.get_profile(),
            }

        # =====================================================
        # MISSING VALUES
        # =====================================================

        if operation == "missing_values":

            missing = working_df.isna().sum()

            missing = {str(k): int(v) for k, v in missing.items() if v > 0}

            return {"success": True, "operation": "missing_values", "result": missing}

        # =====================================================
        # COUNT
        # =====================================================

        if operation == "count":

            return {
                "success": True,
                "operation": "count",
                "result": int(len(working_df)),
            }

        # =====================================================
        # NUMERIC OPERATIONS
        # =====================================================

        if operation in ["average", "sum", "maximum", "minimum"]:

            if column is None:

                return {
                    "success": False,
                    "message": (
                        "I could not identify the "
                        "financial column needed "
                        "for this calculation."
                    ),
                }

            if not pd.api.types.is_numeric_dtype(working_df[column]):

                return {
                    "success": False,
                    "message": (f"'{column}' is not " "a numeric column."),
                }

            series = working_df[column].dropna()

            if len(series) == 0:

                return {
                    "success": False,
                    "message": (f"There is no usable data " f"in '{column}'."),
                }

            if operation == "average":

                value = series.mean()

            elif operation == "sum":

                value = series.sum()

            elif operation == "maximum":

                value = series.max()

            else:

                value = series.min()

            return {
                "success": True,
                "operation": operation,
                "column": column,
                "filter": filter_data,
                "result": float(value),
            }

        # =====================================================
        # RANKING
        # =====================================================

        if operation == "ranking":

            if column is None:

                return {
                    "success": False,
                    "message": ("I could not identify the " "column to rank."),
                }

            if not pd.api.types.is_numeric_dtype(working_df[column]):

                return {
                    "success": False,
                    "message": (f"'{column}' is not numeric " "and cannot be ranked."),
                }

            sort_order = plan.get("sort_order", "descending")

            ascending = sort_order == "ascending"

            # -------------------------------------------------
            # SORT DATA
            # -------------------------------------------------

            sort_order = plan.get("sort_order", "descending")

            ascending = sort_order == "ascending"

            sorted_df = working_df.sort_values(
                by=column,
                ascending=ascending
            )

            # -------------------------------------------------
            # IDENTIFY COMPANY / TICKER COLUMN
            # -------------------------------------------------

            entity_column = None

            preferred_entity_columns = [
                "Company_Name",
                "Company Name",
                "Company",
                "Ticker",
                "Symbol",
                "Stock",
                "Entity",
                "Name",
            ]

            for preferred in preferred_entity_columns:

                actual = self.resolve_column(preferred)

                if actual and actual in sorted_df.columns:

                    entity_column = actual
                    break

            # -------------------------------------------------
            # GET TOP UNIQUE COMPANIES
            # -------------------------------------------------

            if entity_column:

                sorted_df = sorted_df.drop_duplicates(
                    subset=[entity_column],
                    keep="first"
                )

            ranking = sorted_df.head(int(limit))

            #--------------------------------------------------
            # SELECT IDENTIFICATION COLUMNS
            # -------------------------------------------------

            result_columns = []

            preferred_entity_columns = [
                "Company_Name",
                "Company",
                "Company Name",
                "Ticker",
                "Symbol",
                "Stock",
                "Entity",
                "Name",
            ]

            # First prioritize company/entity columns
            for preferred in preferred_entity_columns:

                actual = self.resolve_column(preferred)

                if actual and actual in working_df.columns:

                    if actual not in result_columns:
                        result_columns.append(actual)
                                    # Remove duplicate rows based on the entity information
            if result_columns:

                ranking = ranking.drop_duplicates(
                    subset=result_columns
                )

            # Then add other categorical columns
            for c in working_df.columns:

                if c == column:
                    continue

                if c in result_columns:
                    continue

                if not pd.api.types.is_numeric_dtype(working_df[c]):

                    result_columns.append(c)

                if len(result_columns) >= 3:
                    break

            # Always include the metric
            if column not in result_columns:
                result_columns.append(column)
            

            result = ranking[result_columns].fillna("").to_dict(orient="records")

            return {
                "success": True,
                "operation": "ranking",
                "column": column,
                "result": result,
            }

        # =====================================================
        # COMPARISON
        # =====================================================

        if operation == "comparison":

            requested_columns = plan.get("columns", [])

            if not requested_columns and column:
                requested_columns = [column]

            resolved_columns = []

            for requested_column in requested_columns:

                actual_column = self.resolve_column(requested_column)

                if actual_column is not None:
                    resolved_columns.append(actual_column)

            resolved_columns = list(dict.fromkeys(resolved_columns))

            if not resolved_columns:

                return {
                    "success": False,
                    "message": (
                        "I could not identify the financial " "metrics to compare."
                    ),
                }

            non_numeric = [
                c
                for c in resolved_columns
                if not pd.api.types.is_numeric_dtype(working_df[c])
            ]

            if non_numeric:

                return {
                    "success": False,
                    "message": (
                        "These columns are not numeric and "
                        "cannot be compared: " + ", ".join(non_numeric)
                    ),
                }

            # Compare multiple financial metrics.
            if len(resolved_columns) > 1 and not group_column:

                comparison_rows = []

                for metric in resolved_columns:

                    series = working_df[metric].dropna()

                    if len(series) == 0:
                        continue

                    comparison_rows.append(
                        {
                            "metric": metric,
                            "total": float(series.sum()),
                            "average": float(series.mean()),
                            "minimum": float(series.min()),
                            "maximum": float(series.max()),
                        }
                    )

                if not comparison_rows:

                    return {
                        "success": False,
                        "message": (
                            "There is no usable numeric data "
                            "for the requested comparison."
                        ),
                    }

                return {
                    "success": True,
                    "operation": "comparison",
                    "columns": resolved_columns,
                    "comparison_type": "metrics",
                    "result": comparison_rows,
                }

            # Compare one metric across groups.
            metric = resolved_columns[0]

            if group_column:

                result = (
                    working_df.groupby(group_column)[metric]
                    .agg(["mean", "min", "max", "sum"])
                    .round(4)
                    .reset_index()
                    .to_dict(orient="records")
                )

                return {
                    "success": True,
                    "operation": "comparison",
                    "column": metric,
                    "columns": [metric],
                    "comparison_type": "groups",
                    "group_by": group_column,
                    "result": result,
                }

            # Single metric comparison.
            series = working_df[metric].dropna()

            return {
                "success": True,
                "operation": "comparison",
                "column": metric,
                "columns": [metric],
                "comparison_type": "single",
                "result": {
                    "count": int(series.count()),
                    "total": float(series.sum()),
                    "average": float(series.mean()),
                    "minimum": float(series.min()),
                    "maximum": float(series.max()),
                },
            }

        # =========================================================
    # EXPLAIN VERIFIED RESULT WITH GROQ
    # =========================================================

    def explain_result(self, question, result):

        client = self.get_client()

        verified_result = json.dumps(result, indent=2, default=str)

        prompt = (
            """
You are the final answer generator for a financial CSV analysis system.

The user asked:

"""
            + question
            + """

The Python/Pandas system has already calculated and VERIFIED the result.

VERIFIED RESULT:

"""
            + verified_result
            + """

Your job is to give the user a DIRECT answer to their question.

IMPORTANT:

1. Answer the user's question directly.
2. Do NOT describe the calculation process.
3. Do NOT say things like:
   - "The calculation was performed..."
   - "The analysis looked at..."
   - "The ranking was performed..."
   - "According to the result..."
4. Do NOT explain how Pandas calculated the answer.
5. Do NOT repeat the user's question.
6. Do NOT invent information.
7. Only use information contained in VERIFIED RESULT.
8. If the user asks "which company", give the company name.
9. If the user asks "which ticker", give the ticker.
10. If the user asks for a value, give the value.
11. If the user asks for a comparison, summarize the comparison clearly.
12. Keep the answer concise.
13. Use Markdown when useful.
14. Never provide information that is not present in VERIFIED RESULT.

Examples:

User:
Which company has the highest Close price?

Good answer:
Reliance Industries Ltd. (RELIANCE.NS) has the highest Close price at ₹32,253.60.

Bad answer:
"The ranking was performed on the Close column..."

User:
What is the average Close price?

Good answer:
The average Close price is ₹939.61.

Bad answer:
"The calculation was performed on the Close column..."

User:
Which ticker has the highest Beta?

Good answer:
BPCL.NS has the highest Beta at 0.981.

Bad answer:
"The ranking was performed on the Beta column..."

User:
What is the highest Market Cap?

Good answer:
The highest Market Cap is ₹18.88 trillion.

"""
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=700,
        )

        return response.choices[0].message.content.strip()

    # =========================================================
    # CREATE CHART
    # =========================================================

    def create_chart(self, plan, result):

        if not result.get("success"):
            return None

        operation = plan.get("operation")

        column = result.get("column")

        # =====================================================
        # RANKING CHART
        # =====================================================

        if operation == "ranking":

            data = result.get("result")

            if not isinstance(data, list) or not data:

                return None

            chart_df = pd.DataFrame(data)

            if column not in chart_df.columns:
                return None

            category_columns = [c for c in chart_df.columns if c != column]

            if not category_columns:
                return None

            category = category_columns[0]

            return {
                "type": "bar",
                "data": chart_df,
                "x": category,
                "y": column,
                "title": (f"Top {len(chart_df)} " f"by {column}"),
            }

        # =====================================================
        # COMPARISON CHART
        # =====================================================

        if operation == "comparison":

            data = result.get("result")

            if not isinstance(data, list) or not data:
                return None

            chart_df = pd.DataFrame(data)

            comparison_type = result.get("comparison_type")

            if comparison_type == "metrics":

                if "metric" not in chart_df.columns or "total" not in chart_df.columns:
                    return None

                return {
                    "type": "bar",
                    "data": chart_df,
                    "x": "metric",
                    "y": "total",
                    "title": "Financial Metric Comparison",
                }

            if comparison_type == "groups":

                group_by = result.get("group_by")

                if (
                    not group_by
                    or group_by not in chart_df.columns
                    or "mean" not in chart_df.columns
                ):
                    return None

                return {
                    "type": "bar",
                    "data": chart_df,
                    "x": group_by,
                    "y": "mean",
                    "title": f"Average {column} Comparison",
                }

            return None

        # =====================================================
        # TREND CHART
        # =====================================================

        if operation == "trend":

            data = result.get("result")

            if not isinstance(data, list) or not data:

                return None

            chart_df = pd.DataFrame(data)

            if column not in chart_df.columns:
                return None

            date_columns = [c for c in chart_df.columns if c != column]

            if not date_columns:
                return None

            date_column = date_columns[0]

            return {
                "type": "line",
                "data": chart_df,
                "x": date_column,
                "y": column,
                "title": (f"{column} Trend"),
            }

        return None

    # =========================================================
    # MAIN QUESTION FUNCTION
    # =========================================================

    def answer_question(self, question):

        try:

            # -------------------------------------------------
            # Step 1: Understand question
            # -------------------------------------------------

            plan = self.understand_question(question)

            # -------------------------------------------------
            # Step 2: Execute against actual CSV
            # -------------------------------------------------

            result = self.execute_query(plan)

            # -------------------------------------------------
            # Step 3: Dataset cannot answer
            # -------------------------------------------------

            if not result["success"]:

                return {
                    "answer": (
                        "⚠️ **I can't answer that "
                        "from the uploaded dataset.**\n\n"
                        f"{result['message']}"
                    ),
                    "chart": None,
                }

            # -------------------------------------------------
            # Step 4: Explain verified result
            # -------------------------------------------------

            answer = self.explain_result(question, result)

            # -------------------------------------------------
            # Step 5: Create chart
            # -------------------------------------------------

            chart = self.create_chart(plan, result)

            # -------------------------------------------------
            # Step 6: Return structured response
            # -------------------------------------------------

            return {"answer": answer, "chart": chart}

        except json.JSONDecodeError:

            return {
                "answer": (
                    "⚠️ I couldn't understand "
                    "the question well enough "
                    "to analyze the dataset."
                ),
                "chart": None,
            }

        except Exception as e:

            return {
                "answer": (
                    "⚠️ I couldn't process " "your question.\n\n" f"Error: {str(e)}"
                ),
                "chart": None,
            }
