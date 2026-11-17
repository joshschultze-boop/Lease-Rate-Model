# Preprocessing Script

# Read Excel file

import pandas as pd

def categorical_columns_unknown(df, categorical_columns):
    df = df.copy()
    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna("unknown")
    return df

def convert_data_types(df, categorical_columns, numerical_columns, date_columns,epoch):

    #Categorical Columns
    df = categorical_columns_unknown(df, categorical_columns)
    df[categorical_columns] = df[categorical_columns].astype("category")
    

    #Numerical Columns
    df[numerical_columns] = df[numerical_columns].replace("unknown", pd.NA)
    df[numerical_columns] = df[numerical_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if epoch is None:
        epoch = pd.Timestamp("2000-01-01")

    #Date columns
    df[date_columns] = df[date_columns].replace("unknown", pd.NA)
    for column in date_columns:
        df[column] = (
            pd.to_datetime(df[column], errors="coerce") - epoch
        ).dt.days
    return df


def clean_dataframe(df, exclude_values, required_columns):
    df = df.copy()

    # Drop rows containing excluded values.
    for column, values in exclude_values.items():
        is_excluded = df[column].isin(values)

        # isin() never matches blank cells, so None in the list is handled here.
        if None in values:
            is_excluded = is_excluded | df[column].isna()

        df = df.loc[~is_excluded]

    # Drop rows that are blank or "unknown" in any required column.
    # (convert_data_types fills blank categorical cells with "unknown",
    # so checking for blanks alone would keep those rows.)
    for column in required_columns:
        text = df[column].astype("string").str.strip().str.casefold()
        is_missing = df[column].isna() | text.eq("unknown").fillna(False)
        df = df.loc[~is_missing]

    return df.reset_index(drop=True)

def drop_excluded_columns(df, columns_to_exclude):
    """Remove specified columns from the dataframe."""
    df = df.copy()

    missing_columns = [
        column for column in columns_to_exclude
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    return df.drop(columns=columns_to_exclude)

def remove_rows_above_maximums(df, maximum_values):
    """
    Remove rows where a specified numerical column exceeds its maximum.

    maximum_values example:
        {"age": 120, "income": 1_000_000}
    """
    df = df.copy()

    missing_columns = [
        column for column in maximum_values
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    for column, maximum in maximum_values.items():
        df = df.loc[
            df[column].isna() | df[column].le(maximum)
        ]

    return df.reset_index(drop=True)


def combine_categorical_columns(
    df, columns, new_column, separator=" ", zip_column="Zip"
):
    df = df.copy()

    if isinstance(columns, str):
        columns = [columns]

    if not columns:
        raise ValueError("Provide at least one source column.")

    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    combined = None

    for column in columns:
        values = df[column].astype("string")

        if column == zip_column:
            values = values.str.strip()

            zip_mask = values.str.fullmatch(
                r"[0-9]{1,5}(?:\.0+)?",
                na=False,
            )

            formatted_zip = (
                values.str.replace(r"\.0+$", "", regex=True)
                .str.zfill(5)
            )

            values = values.where(~zip_mask, formatted_zip)

        if combined is None:
            combined = values
        else:
            combined = combined.str.cat(values, sep=separator)

    df[new_column] = combined
    return df

def add_unknown_indicator(df, columns, new_column):
    df = df.copy()

    if isinstance(columns, str):
        columns = [columns]

    if not columns:
        raise ValueError("Provide at least one source column.")

    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    unknown_mask = pd.Series(False, index=df.index)

    for column in columns:
        values = (
            df[column]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        unknown_mask |= (
            values.isna()
            | values.eq("unknown").fillna(False)
        )

    df[new_column] = unknown_mask.astype("int8")
    return df


def _require_columns(df, columns):
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")


def _numeric_values(values):
    return pd.to_numeric(values, errors="coerce")


def add_yard_property_ratios(
    df,
    yard_column="Yard Size (AC)",
    lease_building_column="Lease Building (SF)",
    property_acres_column="Property (AC)",
    property_building_column="Property Building (SF)",
    acres_ratio_column="yard / prop ratio (acres)",
    sf_ratio_column="yard / prop ratio (sf)",
):
    df = df.copy()

    _require_columns(
        df,
        [
            yard_column,
            lease_building_column,
            property_acres_column,
            property_building_column,
        ],
    )

    yard = _numeric_values(df[yard_column])
    lease_building = _numeric_values(df[lease_building_column])
    property_acres = _numeric_values(df[property_acres_column])
    property_building = _numeric_values(df[property_building_column])

    df[acres_ratio_column] = _numeric_values(
        yard.where(yard.ge(0))
        / property_acres.where(property_acres.gt(0))
    ).round(6)

    df[sf_ratio_column] = _numeric_values(
        lease_building.where(lease_building.ge(0))
        / property_building.where(property_building.gt(0))
    ).round(6)

    return df

def add_log_transforms(df, columns, method="log"):
    """
    Add natural-log columns without changing the source columns.

    method="log":   ln(x), valid for x > 0.
    method="log1p": ln(1 + x), valid for x > -1; zero becomes zero.

    Missing, nonnumeric, infinite, and out-of-domain values become NaN.
    """
    import numpy as np
    if isinstance(columns, str):
        columns = [columns]

    if not columns:
        raise ValueError("Provide at least one source column.")

    if method not in {"log", "log1p"}:
        raise ValueError("method must be 'log' or 'log1p'.")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    new_columns = [f"{column}_{method}" for column in columns]
    existing = [column for column in new_columns if column in df.columns]
    if existing:
        raise ValueError(f"Output columns already exist: {existing}")

    df = df.copy()
    transform = np.log if method == "log" else np.log1p
    lower_bound = 0 if method == "log" else -1

    for column, new_column in zip(columns, new_columns):
        values = pd.to_numeric(df[column], errors="coerce").astype("float64")

        # Mask invalid inputs before applying the logarithm.
        valid = np.isfinite(values) & values.gt(lower_bound)
        df[new_column] = transform(values.where(valid))

    return df


# ---------------------------------------------------------------------
# Small steps that were written inline in preprocessing_script.py
# ---------------------------------------------------------------------

def clean_target_columns(df, target_columns):
    """Turn target columns into numbers, removing a leading ' (Excel text marker)."""
    df = df.copy()

    for column in target_columns:
        values = (
            df[column]
            .astype("string")
            .str.strip()
            .str.removeprefix("'")
        )
        df[column] = pd.to_numeric(values, errors="raise")

    return df


def round_columns(df, columns, decimals=2):
    """Round numeric columns to a set number of decimal places."""
    df = df.copy()
    _require_columns(df, columns)

    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="raise").round(decimals)

    return df


def fill_blanks_with_zero(df, columns):
    """Fill blank cells with 0 in the given columns only."""
    df = df.copy()
    _require_columns(df, columns)

    df[columns] = df[columns].fillna(0)
    return df
