import os
import json
import re
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)


class FinancialAnalyzer:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self, df):

        self.df = df.copy()

        # Clean column names
        self.df.columns = (
            self.df.columns
            .astype(str)
            .str.strip()
        )

        # Convert numeric-looking columns
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

        # Column lookup
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
    # GROQ CLIENT
    # =========================================================

    def get_client(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found."
            )

        return Groq(api_key=api_key)

    # =========================================================
    # PROFILE
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
            ),
        }

    # =========================================================
    # STATISTICS
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
                "std": float(series.std()),
            }

        return statistics

    # =========================================================
    # SUMMARY
    # =========================================================

    def get_summary(self):

        summary = {}

        for column in (
            self.df
            .select_dtypes(include="number")
            .columns
        ):

            series = self.df[column].dropna()

            if series.empty:
                continue

            summary[column] = {
                "sum": float(series.sum()),
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

        return summary

    # =========================================================
    # SAMPLE
    # =========================================================

    def get_sample(self, rows=10):

        sample = self.df.head(rows).copy()

        sample = sample.fillna("")

        return sample.to_dict(
            orient="records"
        )

    # =========================================================
    # CATEGORICAL VALUES
    # =========================================================

    def get_categorical_values(
        self,
        max_values=300
    ):

        values = {}

        for column in (
            self.df
            .select_dtypes(exclude="number")
            .columns
        ):

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
    # RESOLVE COLUMN
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
    # FIND ENTITY COLUMN
    # =========================================================

    def find_entity_column(self):

        preferred_columns = [

            # Company names first
            "Company_Name",
            "Company Name",
            "Company",

            # Then ticker
            "Ticker",
            "Symbol",
            "Stock",

            # Generic
            "Entity",
            "Name",
        ]

        for preferred in preferred_columns:

            actual = self.resolve_column(
                preferred
            )

            if (
                actual is not None
                and actual in self.df.columns
            ):

                return actual

        # Automatic fallback
        normalized_columns = {
            column: self.normalize_text(column)
            for column in self.df.columns
        }

        for column, normalized in (
            normalized_columns.items()
        ):

            if (
                "company" in normalized
                or "ticker" in normalized
                or "symbol" in normalized
                or normalized == "stock"
            ):

                return column

        return None

    # =========================================================
    # INFER FINANCIAL COLUMNS
    # =========================================================

    def infer_columns_from_question(
        self,
        question
    ):

        question_normalized = (
            self.normalize_text(question)
        )

        matches = []

        aliases = {

            "revenue": [
                "revenue",
                "revenues",
                "sales",
                "net sales",
                "total revenue",
            ],

            "net profit": [
                "net profit",
                "net income",
                "profit after tax",
                "pat",
                "net earnings",
            ],

            "gross profit": [
                "gross profit"
            ],

            "operating profit": [
                "operating profit",
                "operating income",
            ],

            "expenses": [
                "expenses",
                "expense",
                "total expenses",
                "operating expenses",
            ],

            "assets": [
                "assets",
                "total assets",
            ],

            "liabilities": [
                "liabilities",
                "total liabilities",
            ],

            "equity": [
                "equity",
                "shareholders equity",
                "stockholders equity",
            ],

            "cash flow": [
                "cash flow",
                "cash flows",
            ],

            "ebitda": [
                "ebitda"
            ],

            "market cap": [
                "market cap",
                "market capitalization",
                "market value",
            ],

            "pe ratio": [
                "pe",
                "p e",
                "pe ratio",
                "price earnings",
            ],

            "forward pe": [
                "forward pe",
                "forward p e",
            ],

            "peg ratio": [
                "peg",
                "peg ratio",
            ],

            "price to book": [
                "price to book",
                "p b",
                "pb ratio",
            ],

            "dividend yield": [
                "dividend yield",
            ],

            "eps": [
                "eps",
                "earnings per share",
            ],

            "beta": [
                "beta"
            ],

            "close": [
                "close",
                "closing price",
                "close price",
            ],

            "open": [
                "open",
                "opening price",
            ],

            "high": [
                "high",
                "highest price",
            ],

            "low": [
                "low",
                "lowest price",
            ],

            "volume": [
                "volume",
                "trading volume",
            ],

            "daily return": [
                "daily return",
                "return",
                "daily returns",
            ],

            "volatility": [
                "volatility",
                "volatility 20d",
                "20 day volatility",
            ],

            "market cap": [
                "market cap",
                "market capitalization",
                "market value",
            ],
        }

        # -----------------------------------------------------
        # First: actual column names
        # -----------------------------------------------------

        numeric_columns = (
            self.df
            .select_dtypes(include="number")
            .columns
        )

        for column in numeric_columns:

            normalized_column = (
                self.normalize_text(column)
            )

            if (
                normalized_column
                and normalized_column
                in question_normalized
            ):

                matches.append(column)

        # -----------------------------------------------------
        # Alias matching
        # -----------------------------------------------------

        for canonical, alias_list in (
            aliases.items()
        ):

            alias_found = any(
                self.normalize_text(alias)
                in question_normalized
                for alias in alias_list
            )

            if not alias_found:
                continue

            for column in numeric_columns:

                normalized_column = (
                    self.normalize_text(column)
                )

                if canonical == "revenue":

                    if (
                        "revenue"
                        in normalized_column
                        or "sales"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "net profit":

                    if (
                        "net profit"
                        in normalized_column
                        or "net income"
                        in normalized_column
                        or "profit after tax"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "gross profit":

                    if (
                        "gross profit"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "operating profit":

                    if (
                        "operating profit"
                        in normalized_column
                        or "operating income"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "expenses":

                    if "expense" in normalized_column:
                        matches.append(column)

                elif canonical == "assets":

                    if "asset" in normalized_column:
                        matches.append(column)

                elif canonical == "liabilities":

                    if "liabilit" in normalized_column:
                        matches.append(column)

                elif canonical == "equity":

                    if "equity" in normalized_column:
                        matches.append(column)

                elif canonical == "cash flow":

                    if "cash flow" in normalized_column:
                        matches.append(column)

                elif canonical == "ebitda":

                    if "ebitda" in normalized_column:
                        matches.append(column)

                elif canonical == "market cap":

                    if "market cap" in normalized_column:
                        matches.append(column)

                elif canonical == "pe ratio":

                    if (
                        "pe ratio" in normalized_column
                        or normalized_column == "pe"
                    ):
                        matches.append(column)

                elif canonical == "forward pe":

                    if "forward pe" in normalized_column:
                        matches.append(column)

                elif canonical == "peg ratio":

                    if "peg" in normalized_column:
                        matches.append(column)

                elif canonical == "price to book":

                    if (
                        "price to book"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "dividend yield":

                    if (
                        "dividend yield"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "eps":

                    if (
                        normalized_column == "eps"
                        or "earnings per share"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "beta":

                    if normalized_column == "beta":
                        matches.append(column)

                elif canonical == "close":

                    if (
                        normalized_column == "close"
                        or "close price"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "open":

                    if (
                        normalized_column == "open"
                        or "open price"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "high":

                    if (
                        normalized_column == "high"
                        or "high price"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "low":

                    if (
                        normalized_column == "low"
                        or "low price"
                        in normalized_column
                    ):
                        matches.append(column)

                elif canonical == "volume":

                    if normalized_column == "volume":
                        matches.append(column)

                elif canonical == "daily return":

                    if "daily return" in normalized_column:
                        matches.append(column)

                elif canonical == "volatility":

                    if "volatility" in normalized_column:
                        matches.append(column)

        return list(
            dict.fromkeys(matches)
        )

    # =========================================================
    # VALIDATE FILTER
    # =========================================================

    def validate_filter(
        self,
        filter_data
    ):

        if not filter_data:
            return None

        column_name = filter_data.get(
            "column"
        )

        value = filter_data.get(
            "value"
        )

        if (
            not column_name
            or value is None
        ):
            return None

        actual_column = (
            self.resolve_column(
                column_name
            )
        )

        if actual_column is None:

            return {
                "valid": False,
                "message": (
                    f"The dataset does not contain "
                    f"a column called '{column_name}'."
                ),
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

        # Exact
        matches = series[
            series.str.lower()
            == requested_value.lower()
        ]

        if len(matches) > 0:

            actual_value = matches.iloc[0]

            return {
                "valid": True,
                "column": actual_column,
                "value": actual_value,
            }

        # Partial
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
                    f"'{value}' is not present "
                    "in the uploaded dataset."
                ),
            }

        actual_value = (
            partial_matches.iloc[0]
        )

        return {
            "valid": True,
            "column": actual_column,
            "value": actual_value,
        }

    # =========================================================
    # EXTRACT NUMBER
    # =========================================================

    def extract_limit(self, question):

        q = self.normalize_text(
            question
        )

        patterns = [

            r"\btop\s+(\d+)\b",

            r"\bbottom\s+(\d+)\b",

            r"\bhighest\s+(\d+)\b",

            r"\blowest\s+(\d+)\b",

            r"\blargest\s+(\d+)\b",

            r"\bsmallest\s+(\d+)\b",

            r"\b(\d+)\s+highest\b",

            r"\b(\d+)\s+lowest\b",

            r"\b(\d+)\s+largest\b",

            r"\b(\d+)\s+smallest\b",

            r"\b(\d+)\s+top\b",

            r"\b(\d+)\s+bottom\b",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                q
            )

            if match:

                return int(
                    match.group(1)
                )

        return 1

    # =========================================================
    # DETECT RANKING
    # =========================================================

    def detect_ranking_direction(
        self,
        question
    ):

        q = self.normalize_text(
            question
        )

        lowest_words = [
            "lowest",
            "bottom",
            "smallest",
            "least",
            "minimum",
        ]

        highest_words = [
            "highest",
            "top",
            "largest",
            "greatest",
            "maximum",
        ]

        words = q.split()

        if any(
            word in words
            for word in lowest_words
        ):

            return "ascending"

        if any(
            word in words
            for word in highest_words
        ):

            return "descending"

        return None

    # =========================================================
    # IS RANKING QUESTION?
    # =========================================================

    def is_ranking_question(
        self,
        question
    ):

        q = self.normalize_text(
            question
        )

        ranking_words = [
            "top",
            "bottom",
            "highest",
            "lowest",
            "largest",
            "smallest",
            "maximum",
            "minimum",
            "greatest",
            "least",
        ]

        words = q.split()

        return any(
            word in words
            for word in ranking_words
        )

    # =========================================================
    # UNDERSTAND QUESTION
    # =========================================================

    def understand_question(
        self,
        question
    ):

        # -----------------------------------------------------
        # ALWAYS HANDLE RANKING LOCALLY
        # -----------------------------------------------------

        if self.is_ranking_question(
            question
        ):

            detected_columns = (
                self.infer_columns_from_question(
                    question
                )
            )

            if detected_columns:

                metric_column = (
                    detected_columns[0]
                )

                entity_column = (
                    self.find_entity_column()
                )

                direction = (
                    self.detect_ranking_direction(
                        question
                    )
                )

                limit = (
                    self.extract_limit(
                        question
                    )
                )

                limit = max(
                    1,
                    min(
                        limit,
                        len(self.df)
                    )
                )

                return {
                    "operation": "ranking",
                    "column": metric_column,
                    "columns": [
                        metric_column
                    ],
                    "group_by": entity_column,
                    "filter": None,
                    "limit": limit,
                    "sort_order": (
                        direction
                        or "descending"
                    ),
                }

        # -----------------------------------------------------
        # NON-RANKING QUESTIONS → GROQ
        # -----------------------------------------------------

        client = self.get_client()

        profile = self.get_profile()

        categorical_values = (
            self.get_categorical_values()
        )

        prompt = f"""
You are a query planner for a financial CSV analysis system.

Your ONLY job is to convert the user's question into a JSON query plan.

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

CATEGORICAL VALUES:

{json.dumps(
    categorical_values,
    indent=2,
    default=str
)}

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

Required fields:

operation
column
columns
group_by
filter
limit

RULES:

1. Only use columns that actually exist.

2. Never invent a column.

3. Average means average.

4. Total means sum.

5. Maximum means maximum.

6. Minimum means minimum.

7. If comparing multiple metrics, use comparison.

8. If comparing a metric across entities, use group_by.

9. If the user specifies a company/entity,
   use the actual company/entity column as filter.

10. Questions about the dataset itself mean overview.

11. Questions about missing data mean missing_values.

12. Questions asking how many records exist mean count.

13. Return ONLY JSON.

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
                },
            ],

            temperature=0,

            max_tokens=500,
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        content = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        plan = json.loads(
            content
        )

        # Deterministic fallback
        inferred_columns = (
            self.infer_columns_from_question(
                question
            )
        )

        if (
            not plan.get("column")
            and inferred_columns
        ):

            plan["column"] = (
                inferred_columns[0]
            )

        if (
            not plan.get("columns")
            and inferred_columns
        ):

            plan["columns"] = (
                inferred_columns
            )

        return plan

    # =========================================================
    # APPLY FILTER
    # =========================================================

    def get_working_dataframe(
        self,
        filter_data
    ):

        validated_filter = (
            self.validate_filter(
                filter_data
            )
        )

        if validated_filter:

            if not validated_filter[
                "valid"
            ]:

                return (
                    None,
                    validated_filter[
                        "message"
                    ]
                )

            filter_column = (
                validated_filter[
                    "column"
                ]
            )

            filter_value = (
                validated_filter[
                    "value"
                ]
            )

            working_df = self.df[
                self.df[
                    filter_column
                ]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(
                    filter_value
                )
                .strip()
                .lower()
            ].copy()

        else:

            working_df = self.df.copy()

        if working_df.empty:

            return (
                None,
                "No matching records were found "
                "in the uploaded dataset."
            )

        return (
            working_df,
            None
        )

    # =========================================================
    # EXECUTE QUERY
    # =========================================================

    def execute_query(
        self,
        plan
    ):

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

        # -----------------------------------------------------
        # Resolve metric column
        # -----------------------------------------------------

        column = None

        if column_name:

            column = (
                self.resolve_column(
                    column_name
                )
            )

            if column is None:

                return {
                    "success": False,
                    "message": (
                        f"The column "
                        f"'{column_name}' "
                        "does not exist "
                        "in the dataset."
                    ),
                }

        # -----------------------------------------------------
        # Resolve group column
        # -----------------------------------------------------

        group_column = None

        if group_by:

            group_column = (
                self.resolve_column(
                    group_by
                )
            )

            if group_column is None:

                return {
                    "success": False,
                    "message": (
                        f"The grouping column "
                        f"'{group_by}' "
                        "does not exist."
                    ),
                }

        # -----------------------------------------------------
        # Working dataframe
        # -----------------------------------------------------

        working_df, error = (
            self.get_working_dataframe(
                filter_data
            )
        )

        if error:

            return {
                "success": False,
                "message": error
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
                "result": missing,
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
                ),
            }

        # =====================================================
        # NUMERIC OPERATIONS
        # =====================================================

        if operation in [
            "average",
            "sum",
            "maximum",
            "minimum",
        ]:

            if column is None:

                return {
                    "success": False,
                    "message": (
                        "I could not identify "
                        "the financial column "
                        "needed for this "
                        "calculation."
                    ),
                }

            if not pd.api.types.is_numeric_dtype(
                working_df[column]
            ):

                return {
                    "success": False,
                    "message": (
                        f"'{column}' is not "
                        "a numeric column."
                    ),
                }

            series = (
                working_df[column]
                .dropna()
            )

            if series.empty:

                return {
                    "success": False,
                    "message": (
                        f"There is no usable "
                        f"data in '{column}'."
                    ),
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

            return self.execute_ranking(
                working_df,
                column,
                group_column,
                plan
            )

        # =====================================================
        # COMPARISON
        # =====================================================

        if operation == "comparison":

            requested_columns = (
                plan.get(
                    "columns",
                    []
                )
            )

            if (
                not requested_columns
                and column
            ):

                requested_columns = [
                    column
                ]

            resolved_columns = []

            for requested_column in (
                requested_columns
            ):

                actual_column = (
                    self.resolve_column(
                        requested_column
                    )
                )

                if actual_column:

                    resolved_columns.append(
                        actual_column
                    )

            resolved_columns = list(
                dict.fromkeys(
                    resolved_columns
                )
            )

            if not resolved_columns:

                return {
                    "success": False,
                    "message": (
                        "I could not identify "
                        "the financial metrics "
                        "to compare."
                    ),
                }

            non_numeric = [
                c
                for c in resolved_columns
                if not pd.api.types.is_numeric_dtype(
                    working_df[c]
                )
            ]

            if non_numeric:

                return {
                    "success": False,
                    "message": (
                        "These columns are not "
                        "numeric: "
                        + ", ".join(
                            non_numeric
                        )
                    ),
                }

            # -------------------------------------------------
            # Multiple metrics
            # -------------------------------------------------

            if (
                len(resolved_columns) > 1
                and not group_column
            ):

                comparison_rows = []

                for metric in (
                    resolved_columns
                ):

                    series = (
                        working_df[metric]
                        .dropna()
                    )

                    if series.empty:
                        continue

                    comparison_rows.append(
                        {
                            "metric": metric,
                            "total": float(
                                series.sum()
                            ),
                            "average": float(
                                series.mean()
                            ),
                            "minimum": float(
                                series.min()
                            ),
                            "maximum": float(
                                series.max()
                            ),
                        }
                    )

                return {
                    "success": True,
                    "operation": "comparison",
                    "columns": resolved_columns,
                    "comparison_type": "metrics",
                    "result": comparison_rows,
                }

            # -------------------------------------------------
            # Compare across groups
            # -------------------------------------------------

            metric = resolved_columns[0]

            if group_column:

                result = (
                    working_df
                    .groupby(
                        group_column
                    )[metric]
                    .agg(
                        [
                            "mean",
                            "min",
                            "max",
                            "sum",
                        ]
                    )
                    .round(4)
                    .reset_index()
                    .to_dict(
                        orient="records"
                    )
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

            # -------------------------------------------------
            # Single metric
            # -------------------------------------------------

            series = (
                working_df[metric]
                .dropna()
            )

            return {
                "success": True,
                "operation": "comparison",
                "column": metric,
                "columns": [metric],
                "comparison_type": "single",
                "result": {
                    "count": int(
                        series.count()
                    ),
                    "total": float(
                        series.sum()
                    ),
                    "average": float(
                        series.mean()
                    ),
                    "minimum": float(
                        series.min()
                    ),
                    "maximum": float(
                        series.max()
                    ),
                },
            }

        # =====================================================
        # UNKNOWN OPERATION
        # =====================================================

        return {
            "success": False,
            "message": (
                "I don't know how to "
                "perform that operation "
                "on the dataset."
            ),
        }

    # =========================================================
    # EXECUTE RANKING
    # =========================================================

    def execute_ranking(
        self,
        working_df,
        column,
        group_column,
        plan
    ):

        # -----------------------------------------------------
        # Validate metric
        # -----------------------------------------------------

        if column is None:

            return {
                "success": False,
                "message": (
                    "I could not identify "
                    "the column to rank."
                ),
            }

        if not pd.api.types.is_numeric_dtype(
            working_df[column]
        ):

            return {
                "success": False,
                "message": (
                    f"'{column}' is not numeric "
                    "and cannot be ranked."
                ),
            }

        # -----------------------------------------------------
        # Automatically find entity
        # -----------------------------------------------------

        if group_column is None:

            group_column = (
                self.find_entity_column()
            )

        # -----------------------------------------------------
        # Sort direction
        # -----------------------------------------------------

        sort_order = plan.get(
            "sort_order",
            "descending"
        )

        ascending = (
            sort_order == "ascending"
        )

        # -----------------------------------------------------
        # LIMIT
        # -----------------------------------------------------

        try:

            limit = int(
                plan.get(
                    "limit",
                    10
                )
            )

        except (
            TypeError,
            ValueError
        ):

            limit = 10

        # =====================================================
        # IMPORTANT:
        # COMPANY-LEVEL RANKING
        # =====================================================

        if group_column:

            # Remove rows without company
            ranking_df = (
                working_df[
                    working_df[
                        group_column
                    ].notna()
                ].copy()
            )

            if ranking_df.empty:

                return {
                    "success": False,
                    "message": (
                        "No valid company/entity "
                        "values were found."
                    ),
                }

            # -------------------------------------------------
            # Convert metric to numeric
            # -------------------------------------------------

            ranking_df[column] = pd.to_numeric(
                ranking_df[column],
                errors="coerce"
            )

            ranking_df = (
                ranking_df[
                    ranking_df[column]
                    .notna()
                ]
                .copy()
            )

            # -------------------------------------------------
            # THIS IS THE IMPORTANT FIX
            #
            # If multiple rows exist for the same company,
            # aggregate them first.
            #
            # For Market Cap / PE / Beta / prices etc.,
            # the latest available record is usually the
            # most meaningful value.
            # -------------------------------------------------

            date_column = self.find_date_column()

            if date_column:

                try:

                    ranking_df["_ranking_date"] = (
                        pd.to_datetime(
                            ranking_df[
                                date_column
                            ],
                            errors="coerce"
                        )
                    )

                    ranking_df = (
                        ranking_df
                        .sort_values(
                            "_ranking_date"
                        )
                        .drop_duplicates(
                            subset=[
                                group_column
                            ],
                            keep="last"
                        )
                    )

                    ranking_df = (
                        ranking_df
                        .drop(
                            columns=[
                                "_ranking_date"
                            ],
                            errors="ignore"
                        )
                    )

                except Exception:

                    ranking_df = (
                        ranking_df
                        .sort_values(
                            by=column,
                            ascending=ascending,
                            na_position="last"
                        )
                        .drop_duplicates(
                            subset=[
                                group_column
                            ],
                            keep="first"
                        )
                    )

            else:

                # No date available.
                #
                # Aggregate duplicate companies by MEAN.
                #
                # This prevents the same company from appearing
                # multiple times and makes ranking deterministic.

                ranking_df = (
                    ranking_df
                    .groupby(
                        group_column,
                        as_index=False
                    )[column]
                    .mean()
                )

            # -------------------------------------------------
            # FINAL SORT
            # -------------------------------------------------

            ranking_df = (
                ranking_df
                .sort_values(
                    by=column,
                    ascending=ascending,
                    na_position="last"
                )
                .reset_index(
                    drop=True
                )
            )

            # -------------------------------------------------
            # Limit
            # -------------------------------------------------

            ranking_df = ranking_df.head(
                limit
            )

            # -------------------------------------------------
            # Result columns
            # -------------------------------------------------

            result_columns = [
                group_column,
                column
            ]

            # Add ticker if company name is entity
            ticker_column = (
                self.find_ticker_column()
            )

            if (
                ticker_column
                and ticker_column
                not in result_columns
                and ticker_column
                in ranking_df.columns
            ):

                result_columns.append(
                    ticker_column
                )

            ranking_df = ranking_df[
                result_columns
            ]

        # =====================================================
        # NO ENTITY COLUMN
        # =====================================================

        else:

            ranking_df = (
                working_df[
                    [
                        column
                    ]
                ]
                .dropna()
                .sort_values(
                    by=column,
                    ascending=ascending
                )
                .head(limit)
                .copy()
            )

            result_columns = [
                column
            ]

        # -----------------------------------------------------
        # Convert result
        # -----------------------------------------------------

        result = (
            ranking_df
            .fillna("")
            .to_dict(
                orient="records"
            )
        )

        return {
            "success": True,
            "operation": "ranking",
            "column": column,
            "sort_order": sort_order,
            "limit": limit,
            "entity_column": group_column,
            "result": result,
        }

    # =========================================================
    # FIND DATE COLUMN
    # =========================================================

    def find_date_column(self):

        preferred = [
            "Date",
            "date",
            "Datetime",
            "datetime",
            "Timestamp",
            "timestamp",
        ]

        for name in preferred:

            actual = self.resolve_column(
                name
            )

            if (
                actual
                and actual in self.df.columns
            ):

                return actual

        for column in self.df.columns:

            normalized = (
                self.normalize_text(
                    column
                )
            )

            if (
                "date" in normalized
                or "time" in normalized
            ):

                return column

        return None

    # =========================================================
    # FIND TICKER COLUMN
    # =========================================================

    def find_ticker_column(self):

        preferred = [
            "Ticker",
            "ticker",
            "Symbol",
            "symbol",
        ]

        for name in preferred:

            actual = self.resolve_column(
                name
            )

            if (
                actual
                and actual in self.df.columns
            ):

                return actual

        return None

    # =========================================================
    # EXPLAIN VERIFIED RESULT
    # =========================================================

    def explain_result(
        self,
        question,
        result
    ):

        client = self.get_client()

        verified_result = json.dumps(
            result,
            indent=2,
            default=str
        )

        prompt = f"""
You are the final answer generator for a financial CSV analysis system.

The user asked:

{question}

Python/Pandas has already calculated and VERIFIED the result.

VERIFIED RESULT:

{verified_result}

Answer the user's question directly.

RULES:

1. Only use information inside VERIFIED RESULT.

2. Never invent values.

3. Never change numbers.

4. Never change company names.

5. If the user asks for TOP N, list all N results.

6. If the user asks for LOWEST N or BOTTOM N,
   list all N results.

7. Preserve the ranking order exactly as provided.

8. If the result contains a company/entity and ticker,
   show both when useful.

9. Keep the answer concise.

10. Use Markdown.

11. Do not explain Pandas.

12. Do not describe the calculation process.

13. Do not say "according to the analysis".

14. Do not repeat the question.

15. Never add companies that are not in VERIFIED RESULT.
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
                },
            ],

            temperature=0,

            max_tokens=1000,
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

        if not result.get(
            "success"
        ):

            return None

        operation = (
            result.get(
                "operation"
            )
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

            if (
                not isinstance(
                    data,
                    list
                )
                or not data
            ):

                return None

            chart_df = pd.DataFrame(
                data
            )

            if (
                column
                not in chart_df.columns
            ):

                return None

            entity_column = (
                result.get(
                    "entity_column"
                )
            )

            if (
                not entity_column
                or entity_column
                not in chart_df.columns
            ):

                return None

            direction = (
                result.get(
                    "sort_order"
                )
            )

            if direction == "ascending":

                title = (
                    f"Bottom {len(chart_df)} "
                    f"by {column}"
                )

            else:

                title = (
                    f"Top {len(chart_df)} "
                    f"by {column}"
                )

            return {
                "type": "bar",
                "data": chart_df,
                "x": entity_column,
                "y": column,
                "title": title,
            }

        # =====================================================
        # COMPARISON
        # =====================================================

        if operation == "comparison":

            data = result.get(
                "result"
            )

            if (
                not isinstance(
                    data,
                    list
                )
                or not data
            ):

                return None

            chart_df = pd.DataFrame(
                data
            )

            comparison_type = (
                result.get(
                    "comparison_type"
                )
            )

            if (
                comparison_type
                == "metrics"
            ):

                if (
                    "metric"
                    not in chart_df.columns
                    or "total"
                    not in chart_df.columns
                ):

                    return None

                return {
                    "type": "bar",
                    "data": chart_df,
                    "x": "metric",
                    "y": "total",
                    "title": (
                        "Financial Metric "
                        "Comparison"
                    ),
                }

            if (
                comparison_type
                == "groups"
            ):

                group_by = (
                    result.get(
                        "group_by"
                    )
                )

                if (
                    not group_by
                    or group_by
                    not in chart_df.columns
                    or "mean"
                    not in chart_df.columns
                ):

                    return None

                return {
                    "type": "bar",
                    "data": chart_df,
                    "x": group_by,
                    "y": "mean",
                    "title": (
                        f"Average {column} "
                        "Comparison"
                    ),
                }

        return None

    # =========================================================
    # MAIN ANSWER FUNCTION
    # =========================================================

    def answer_question(
        self,
        question
    ):

        try:

            question = (
                str(question)
                .strip()
            )

            if not question:

                return {
                    "answer": (
                        "Please enter a question."
                    ),
                    "chart": None,
                }

            # -------------------------------------------------
            # STEP 1
            # Understand question
            # -------------------------------------------------

            plan = (
                self.understand_question(
                    question
                )
            )

            # -------------------------------------------------
            # STEP 2
            # Execute with Pandas
            # -------------------------------------------------

            result = (
                self.execute_query(
                    plan
                )
            )

            # -------------------------------------------------
            # STEP 3
            # Error
            # -------------------------------------------------

            if not result.get(
                "success"
            ):

                return {
                    "answer": (
                        "⚠️ **I can't answer that "
                        "from the uploaded dataset.**\n\n"
                        f"{result.get('message', '')}"
                    ),
                    "chart": None,
                }

            # -------------------------------------------------
            # STEP 4
            # Generate explanation
            # -------------------------------------------------

            answer = (
                self.explain_result(
                    question,
                    result
                )
            )

            # -------------------------------------------------
            # STEP 5
            # Chart
            # -------------------------------------------------

            chart = (
                self.create_chart(
                    plan,
                    result
                )
            )

            # -------------------------------------------------
            # STEP 6
            # Return
            # -------------------------------------------------

            return {
                "answer": answer,
                "chart": chart,
            }

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
                    "⚠️ I couldn't process "
                    "your question.\n\n"
                    f"Error: {str(e)}"
                ),
                "chart": None,
            }