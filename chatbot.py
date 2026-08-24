import os
import json
import re
import pandas as pd
from groq import Groq


class FinancialAnalyzer:

    def __init__(self, df):

        self.df = df.copy()

        # =====================================================
        # CLEAN COLUMN NAMES
        # =====================================================

        self.df.columns = (
            self.df.columns
            .astype(str)
            .str.strip()
        )

        # =====================================================
        # CONVERT NUMERIC-LOOKING COLUMNS
        # =====================================================

        for column in self.df.columns:

            converted = pd.to_numeric(
                self.df[column],
                errors="coerce"
            )

            if len(self.df) > 0:

                conversion_ratio = (
                    converted.notna().sum()
                    / len(self.df)
                )

                if conversion_ratio >= 0.5:
                    self.df[column] = converted

        # =====================================================
        # COLUMN LOOKUP
        # =====================================================

        self.column_lookup = {
            self.normalize_text(column): column
            for column in self.df.columns
        }

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_text(text):

        return re.sub(
            r"[^a-z0-9]+",
            " ",
            str(text).lower()
        ).strip()

    # =========================================================
    # DATASET PROFILE
    # =========================================================

    def get_profile(self):

        numeric_columns = (
            self.df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        categorical_columns = (
            self.df
            .select_dtypes(exclude="number")
            .columns
            .tolist()
        )

        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": int(
                self.df.isna().sum().sum()
            ),
            "duplicate_rows": int(
                self.df.duplicated().sum()
            )
        }

    # =========================================================
    # FINANCIAL STATISTICS
    # =========================================================

    def get_statistics(self):

        statistics = {}

        numeric_columns = (
            self.df
            .select_dtypes(include="number")
            .columns
        )

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
                "std": float(series.std())
            }

        return statistics

    # =========================================================
    # DATASET SUMMARY
    # =========================================================

    def get_summary(self):

        summary = {}

        for column in self.df.select_dtypes(
            include="number"
        ).columns:

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

        return sample.to_dict(
            orient="records"
        )

    # =========================================================
    # GET CATEGORICAL VALUES
    # =========================================================

    def get_categorical_values(self, max_values=300):

        values = {}

        for column in self.df.select_dtypes(
            exclude="number"
        ).columns:

            unique_values = (
                self.df[column]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            if len(unique_values) <= max_values:

                values[column] = unique_values

        return values

    # =========================================================
    # GROQ CLIENT
    # =========================================================

    def get_client(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY was not found."
            )

        return Groq(
            api_key=api_key
        )

    # =========================================================
    # FIND COLUMN
    # =========================================================

    def resolve_column(self, column_name):

        if not column_name:
            return None

        normalized = self.normalize_text(
            column_name
        )

        # Exact match
        if normalized in self.column_lookup:

            return self.column_lookup[
                normalized
            ]

        # Partial match
        for key, actual_column in (
            self.column_lookup.items()
        ):

            if (
                normalized in key
                or key in normalized
            ):

                return actual_column

        return None

    # =========================================================
    # VALIDATE ENTITY / FILTER
    # =========================================================

    def validate_filter(self, filter_data):

        if not filter_data:
            return None

        column_name = filter_data.get(
            "column"
        )

        value = filter_data.get(
            "value"
        )

        if not column_name or value is None:
            return None

        actual_column = self.resolve_column(
            column_name
        )

        if actual_column is None:

            return {
                "valid": False,
                "message": (
                    f"The dataset does not contain "
                    f"a column called '{column_name}'."
                )
            }

        series = (
            self.df[actual_column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        requested_value = str(
            value
        ).strip()

        # Exact match
        matches = series[
            series.str.lower()
            == requested_value.lower()
        ]

        if len(matches) == 0:

            # Partial match
            partial_matches = series[
                series.str.lower().str.contains(
                    requested_value.lower(),
                    regex=False
                )
            ]

            if len(partial_matches) == 0:

                return {
                    "valid": False,
                    "message": (
                        f"'{value}' is not present in "
                        f"the uploaded dataset."
                    )
                }

            actual_value = (
                partial_matches.iloc[0]
            )

        else:

            actual_value = matches.iloc[0]

        return {
            "valid": True,
            "column": actual_column,
            "value": actual_value
        }

    # =========================================================
    # UNDERSTAND QUESTION
    # =========================================================

    def understand_question(self, question):

        client = self.get_client()

        profile = self.get_profile()

        categorical_values = (
            self.get_categorical_values()
        )

        prompt = f"""
You are a query planner for a financial CSV
analysis system.

Your ONLY job is to convert the user's question
into JSON.

You MUST NOT answer the question.

AVAILABLE COLUMNS:

{json.dumps(
    profile["column_names"],
    indent=2
)}

NUMERIC COLUMNS:

{json.dumps(
    profile["numeric_columns"],
    indent=2
)}

CATEGORICAL VALUES AVAILABLE IN DATASET:

{json.dumps(
    categorical_values,
    indent=2,
    default=str
)}

ALLOWED OPERATIONS:

- overview
- missing_values
- average
- sum
- maximum
- minimum
- ranking
- comparison
- trend
- count

RETURN ONLY VALID JSON.

Required structure:

{{
    "operation": "...",
    "column": null,
    "group_by": null,
    "filter": null,
    "limit": 10
}}

For an entity/company question:

"filter": {{
    "column": "actual column name",
    "value": "actual dataset value"
}}

IMPORTANT RULES:

1. Only select columns that actually exist.
2. Never invent a company/entity.
3. If the requested company/entity does not
   appear in the dataset, put the requested
   entity in the filter so the application
   can reject it.
4. Use exact column names whenever possible.
5. "highest" → maximum or ranking.
6. "lowest" → minimum or ranking.
7. "average" → average.
8. "total" → sum.
9. "top N" → ranking with limit N.
10. Comparisons → comparison.
11. Trends → trend.
12. Dataset information → overview.
13. Missing data → missing_values.
14. Do not add explanations outside JSON.

USER QUESTION:

{question}
"""

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=0,

            max_tokens=500
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        # Remove markdown fences
        content = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(content)

    # =========================================================
    # EXECUTE QUESTION WITH PANDAS
    # =========================================================

    def execute_query(self, plan):

        operation = plan.get(
            "operation"
        )

        column_name = plan.get(
            "column"
        )

        group_by = plan.get(
            "group_by"
        )

        filter_data = plan.get(
            "filter"
        )

        limit = plan.get(
            "limit",
            10
        )

        # =====================================================
        # VALIDATE COLUMN
        # =====================================================

        column = None

        if column_name:

            column = self.resolve_column(
                column_name
            )

            if column is None:

                return {
                    "success": False,
                    "message": (
                        f"The column '{column_name}' "
                        "does not exist in the dataset."
                    )
                }

        # =====================================================
        # VALIDATE GROUP COLUMN
        # =====================================================

        group_column = None

        if group_by:

            group_column = self.resolve_column(
                group_by
            )

            if group_column is None:

                return {
                    "success": False,
                    "message": (
                        f"The grouping column "
                        f"'{group_by}' does not exist."
                    )
                }

        # =====================================================
        # VALIDATE FILTER
        # =====================================================

        validated_filter = (
            self.validate_filter(
                filter_data
            )
        )

        if validated_filter:

            if not validated_filter["valid"]:

                return {
                    "success": False,
                    "message": (
                        validated_filter["message"]
                    )
                }

            filter_column = (
                validated_filter["column"]
            )

            filter_value = (
                validated_filter["value"]
            )

            working_df = self.df[
                self.df[filter_column]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(filter_value)
                .strip()
                .lower()
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
                    "No matching records were found "
                    "in the uploaded dataset."
                )
            }

        # =====================================================
        # OVERVIEW
        # =====================================================

        if operation == "overview":

            return {
                "success": True,
                "operation": "overview",
                "result": self.get_profile()
            }

        # =====================================================
        # MISSING VALUES
        # =====================================================

        if operation == "missing_values":

            missing = (
                working_df
                .isna()
                .sum()
            )

            missing = {
                str(k): int(v)
                for k, v in missing.items()
                if v > 0
            }

            return {
                "success": True,
                "operation": "missing_values",
                "result": missing
            }

        # =====================================================
        # COUNT
        # =====================================================

        if operation == "count":

            return {
                "success": True,
                "operation": "count",
                "result": int(
                    len(working_df)
                )
            }

        # =====================================================
        # NUMERIC OPERATIONS
        # =====================================================

        if operation in [
            "average",
            "sum",
            "maximum",
            "minimum"
        ]:

            if column is None:

                return {
                    "success": False,
                    "message": (
                        "I could not identify the "
                        "financial column needed "
                        "for this calculation."
                    )
                }

            if not pd.api.types.is_numeric_dtype(
                working_df[column]
            ):

                return {
                    "success": False,
                    "message": (
                        f"'{column}' is not "
                        "a numeric column."
                    )
                }

            series = (
                working_df[column]
                .dropna()
            )

            if len(series) == 0:

                return {
                    "success": False,
                    "message": (
                        f"There is no usable data "
                        f"in '{column}'."
                    )
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
                "result": float(value)
            }

        # =====================================================
        # RANKING
        # =====================================================

        if operation == "ranking":

            if column is None:

                return {
                    "success": False,
                    "message": (
                        "I could not identify the "
                        "column to rank."
                    )
                }

            if not pd.api.types.is_numeric_dtype(
                working_df[column]
            ):

                return {
                    "success": False,
                    "message": (
                        f"'{column}' is not numeric "
                        "and cannot be ranked."
                    )
                }

            ranking = (
                working_df
                .sort_values(
                    by=column,
                    ascending=False
                )
                .head(int(limit))
            )

            result_columns = []

            # Add categorical columns
            for c in working_df.columns:

                if (
                    c != column
                    and not pd.api.types.is_numeric_dtype(
                        working_df[c]
                    )
                ):

                    result_columns.append(c)

                    if len(result_columns) >= 2:
                        break

            # Always include metric
            result_columns.append(column)

            result = (
                ranking[
                    result_columns
                ]
                .fillna("")
                .to_dict(
                    orient="records"
                )
            )

            return {
                "success": True,
                "operation": "ranking",
                "column": column,
                "result": result
            }

        # =====================================================
        # COMPARISON
        # =====================================================

        if operation == "comparison":

            if column is None:

                return {
                    "success": False,
                    "message": (
                        "I could not identify "
                        "the comparison metric."
                    )
                }

            if not pd.api.types.is_numeric_dtype(
                working_df[column]
            ):

                return {
                    "success": False,
                    "message": (
                        f"'{column}' is not numeric."
                    )
                }

            if group_column:

                result = (
                    working_df
                    .groupby(
                        group_column
                    )[column]
                    .agg(
                        [
                            "mean",
                            "min",
                            "max"
                        ]
                    )
                    .round(4)
                    .reset_index()
                    .to_dict(
                        orient="records"
                    )
                )

            else:

                result = {
                    "count": int(
                        working_df[column]
                        .count()
                    ),
                    "average": float(
                        working_df[column]
                        .mean()
                    ),
                    "minimum": float(
                        working_df[column]
                        .min()
                    ),
                    "maximum": float(
                        working_df[column]
                        .max()
                    )
                }

            return {
                "success": True,
                "operation": "comparison",
                "column": column,
                "result": result
            }

        # =====================================================
        # TREND
        # =====================================================

        if operation == "trend":

            if column is None:

                return {
                    "success": False,
                    "message": (
                        "I could not identify "
                        "the metric for the trend."
                    )
                }

            if not pd.api.types.is_numeric_dtype(
                working_df[column]
            ):

                return {
                    "success": False,
                    "message": (
                        f"'{column}' is not numeric."
                    )
                }

            date_columns = []

            for c in working_df.columns:

                normalized = (
                    self.normalize_text(c)
                )

                if (
                    "date" in normalized
                    or "year" in normalized
                    or "time" in normalized
                ):

                    date_columns.append(c)

            if not date_columns:

                return {
                    "success": False,
                    "message": (
                        "No date or year column "
                        "was found in the dataset."
                    )
                }

            date_column = date_columns[0]

            temp = working_df[
                [date_column, column]
            ].copy()

            temp[date_column] = pd.to_datetime(
                temp[date_column],
                errors="coerce"
            )

            temp = temp.dropna(
                subset=[
                    date_column,
                    column
                ]
            )

            if temp.empty:

                return {
                    "success": False,
                    "message": (
                        "The date column could not "
                        "be interpreted."
                    )
                }

            trend = (
                temp
                .sort_values(
                    date_column
                )
                .groupby(
                    date_column
                )[column]
                .mean()
                .reset_index()
                .tail(30)
            )

            trend[date_column] = (
                trend[date_column]
                .astype(str)
            )

            return {
                "success": True,
                "operation": "trend",
                "column": column,
                "result": trend.to_dict(
                    orient="records"
                )
            }

        # =====================================================
        # UNKNOWN OPERATION
        # =====================================================

        return {
            "success": False,
            "message": (
                "I couldn't determine how to "
                "analyze that question."
            )
        }

    # =========================================================
    # EXPLAIN VERIFIED RESULT WITH GROQ
    # =========================================================

    def explain_result(
        self,
        question,
        result
    ):

        client = self.get_client()

        prompt = f"""
You are a financial data analyst.

The user asked:

{question}

The application performed the calculation
directly against the uploaded CSV using Pandas.

VERIFIED RESULT:

{json.dumps(
    result,
    indent=2,
    default=str
)}

Your job is ONLY to explain this verified result.

STRICT RULES:

1. Never invent numbers.
2. Never introduce companies that are not
   in the result.
3. Never use outside financial knowledge.
4. Do not contradict the verified result.
5. If the result says an entity is unavailable,
   say so.
6. Keep the answer concise and professional.
7. Mention the relevant column when useful.
8. Format large numbers clearly.
"""

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=0,

            max_tokens=700
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    # =========================================================
    # CREATE CHART
    # =========================================================

    def create_chart(
        self,
        plan,
        result
    ):

        if not result.get("success"):
            return None

        operation = plan.get(
            "operation"
        )

        column = result.get(
            "column"
        )

        # =====================================================
        # RANKING CHART
        # =====================================================

        if operation == "ranking":

            data = result.get(
                "result"
            )

            if not isinstance(
                data,
                list
            ) or not data:

                return None

            chart_df = pd.DataFrame(
                data
            )

            if column not in chart_df.columns:
                return None

            category_columns = [
                c
                for c in chart_df.columns
                if c != column
            ]

            if not category_columns:
                return None

            category = category_columns[0]

            return {
                "type": "bar",
                "data": chart_df,
                "x": category,
                "y": column,
                "title": (
                    f"Top {len(chart_df)} "
                    f"by {column}"
                )
            }

        # =====================================================
        # COMPARISON CHART
        # =====================================================

        if operation == "comparison":

            data = result.get(
                "result"
            )

            if not isinstance(
                data,
                list
            ) or not data:

                return None

            chart_df = pd.DataFrame(
                data
            )

            category_columns = [
                c
                for c in chart_df.columns
                if c not in [
                    "mean",
                    "min",
                    "max"
                ]
            ]

            if not category_columns:
                return None

            if "mean" not in chart_df.columns:
                return None

            category = category_columns[0]

            return {
                "type": "bar",
                "data": chart_df,
                "x": category,
                "y": "mean",
                "title": (
                    f"Average {column} "
                    "Comparison"
                )
            }

        # =====================================================
        # TREND CHART
        # =====================================================

        if operation == "trend":

            data = result.get(
                "result"
            )

            if not isinstance(
                data,
                list
            ) or not data:

                return None

            chart_df = pd.DataFrame(
                data
            )

            if column not in chart_df.columns:
                return None

            date_columns = [
                c
                for c in chart_df.columns
                if c != column
            ]

            if not date_columns:
                return None

            date_column = date_columns[0]

            return {
                "type": "line",
                "data": chart_df,
                "x": date_column,
                "y": column,
                "title": (
                    f"{column} Trend"
                )
            }

        return None

    # =========================================================
    # MAIN QUESTION FUNCTION
    # =========================================================

    def answer_question(
        self,
        question
    ):

        try:

            # -------------------------------------------------
            # Step 1: Understand question
            # -------------------------------------------------

            plan = self.understand_question(
                question
            )

            # -------------------------------------------------
            # Step 2: Execute against actual CSV
            # -------------------------------------------------

            result = self.execute_query(
                plan
            )

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
                    "chart": None
                }

            # -------------------------------------------------
            # Step 4: Explain verified result
            # -------------------------------------------------

            answer = self.explain_result(
                question,
                result
            )

            # -------------------------------------------------
            # Step 5: Create chart
            # -------------------------------------------------

            chart = self.create_chart(
                plan,
                result
            )

            # -------------------------------------------------
            # Step 6: Return structured response
            # -------------------------------------------------

            return {
                "answer": answer,
                "chart": chart
            }

        except json.JSONDecodeError:

            return {
                "answer": (
                    "⚠️ I couldn't understand "
                    "the question well enough "
                    "to analyze the dataset."
                ),
                "chart": None
            }

        except Exception as e:

            return {
                "answer": (
                    "⚠️ I couldn't process "
                    "your question.\n\n"
                    f"Error: {str(e)}"
                ),
                "chart": None
            }