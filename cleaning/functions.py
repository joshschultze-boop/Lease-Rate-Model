# cleaning/functions.py  (your cleaning_script_functions.py, unchanged; what each cleaning step does)

import math
import re
from copy import copy
from numbers import Real
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter


# ---------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------

STREET_ABBREVIATIONS = {
    "Street": "St",
    "Avenue": "Ave",
    "Boulevard": "Blvd",
    "Road": "Rd",
    "Drive": "Dr",
    "Lane": "Ln",
    "Court": "Ct",
    "Circle": "Cir",
    "Terrace": "Ter",
    "Place": "Pl",
    "Square": "Sq",
    "Trail": "Trl",
    "Parkway": "Pkwy",
    "Commons": "Cmns",
}

STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR",
}

ACRE_SF = 43560
METERS_PER_MILE = 1609.344


def _load_workbook(path):
    """Open a workbook, keeping macros when it is an .xlsm file."""
    path = Path(path)
    return load_workbook(path, keep_vba=path.suffix.lower() == ".xlsm")


def _get_worksheet(workbook, sheet_name=None, default_first=False):
    """Return the named sheet, or the active/first sheet if none is given."""
    if sheet_name is None:
        return workbook.worksheets[0] if default_first else workbook.active

    if sheet_name not in workbook.sheetnames:
        raise ValueError(
            f"Worksheet '{sheet_name}' was not found. "
            f"Available worksheets: {workbook.sheetnames}"
        )

    return workbook[sheet_name]


def _header_columns(worksheet):
    """Match each header in row 1 with its Excel column number."""
    return {
        str(cell.value).strip(): cell.column
        for cell in worksheet[1]
        if cell.value is not None
    }


def _get_or_create_column(worksheet, header_columns, column_name):
    """Return a column number, adding the header at the end if needed."""
    if column_name not in header_columns:
        column_number = worksheet.max_column + 1
        worksheet.cell(row=1, column=column_number, value=column_name)
        header_columns[column_name] = column_number

    return header_columns[column_name]


def _require_columns(header_columns, required):
    missing = [name for name in required if name not in header_columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def _is_blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _is_number(value):
    return isinstance(value, Real) and not isinstance(value, bool)


def _clean_text(value):
    """Strip text values and turn blank text into None."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _save(workbook, output_excel_file):
    output_path = Path(output_excel_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


# ---------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------

def address_abbreviator(input_excel_file,
                        column_names,
                        output_excel_file,
                        sheet_name=None):
    """Abbreviate street types (Street -> St, etc.) in the given columns."""
    column_names = _as_list(column_names)

    patterns = [
        (re.compile(rf"\b{re.escape(full_name)}\b", re.IGNORECASE), abbreviation)
        for full_name, abbreviation in STREET_ABBREVIATIONS.items()
    ]

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)

        for column_name in column_names:
            column_number = header_columns.get(column_name.strip())

            # Skip columns that are not present.
            if column_number is None:
                continue

            for row_number in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row_number, column=column_number)

                # Do not modify formulas or non-text values.
                if cell.data_type == "f" or not isinstance(cell.value, str):
                    continue

                updated_address = cell.value
                for pattern, abbreviation in patterns:
                    updated_address = pattern.sub(abbreviation, updated_address)

                cell.value = updated_address

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


# Old (misspelled) name kept so any existing calls still work.
address_abbreviatior = address_abbreviator


def clean_column(input_excel_file,
                 column_names,
                 output_excel_file,
                 sheet_name=None):
    """Strip and uppercase text in the given columns."""
    column_names = _as_list(column_names)

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)

        for column_name in column_names:
            column_number = header_columns.get(column_name.strip())

            # Skip columns that are not present.
            if column_number is None:
                continue

            for row_number in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row_number, column=column_number)

                # Do not modify formulas.
                if cell.data_type == "f":
                    continue

                # Preserve blanks, numbers, and dates.
                if isinstance(cell.value, str):
                    cell.value = cell.value.strip().upper()

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


# ---------------------------------------------------------------------
# Addresses and geocoding
# ---------------------------------------------------------------------

def fill_addresses(input_excel_file,
                   output_excel_file,
                   sheet_name=None):
    """
    Build "Full Address" from the address columns.

    Address columns MUST be named "Street Address", "City", "State", "Zip".
    """
    address_columns = ("Street Address", "City", "State", "Zip")

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, address_columns)

        full_address_column = _get_or_create_column(
            worksheet, header_columns, "Full Address"
        )

        for row_number in range(2, worksheet.max_row + 1):
            street, city, state, zip_code = (
                _clean_text(
                    worksheet.cell(row=row_number, column=header_columns[name]).value
                )
                for name in address_columns
            )

            if _is_number(zip_code):
                zip_code = str(int(zip_code)).zfill(5)
            elif zip_code is not None:
                zip_code = str(zip_code)

            if all([street, city, state, zip_code]):
                worksheet.cell(
                    row=row_number,
                    column=full_address_column,
                ).value = f"{street}, {city}, {state} {zip_code}"

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


def _valid_coordinates(latitude, longitude):
    """Return numeric coordinates, or None if they are invalid."""
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return None

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return None

    return latitude, longitude


def _format_script_address(address):
    """Format an address returned by Nominatim."""
    street = " ".join(
        str(part).strip()
        for part in [address.get("house_number"), address.get("road")]
        if part
    )

    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or ""
    )

    state = str(address.get("state") or "").strip()

    # Nominatim usually returns values such as "US-CO".
    state_code = str(address.get("ISO3166-2-lvl4") or "").strip()
    if state_code.startswith("US-"):
        state = state_code.removeprefix("US-")

    zip_code = str(address.get("postcode") or "").strip()

    state_zip = " ".join(part for part in [state, zip_code] if part)

    return ", ".join(part for part in [street, city, state_zip] if part)


def geocode_address(
    input_excel_file,
    output_excel_file,
    sheet_name=None,
    generate_script_address=False,
):
    """
    Fill Latitude and Longitude using Full Address.

    If generate_script_address is True, also fill:
        - Script Address
        - County
    """
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, ["Full Address"])

        full_address_column = header_columns["Full Address"]
        latitude_column = _get_or_create_column(worksheet, header_columns, "Latitude")
        longitude_column = _get_or_create_column(worksheet, header_columns, "Longitude")

        if generate_script_address:
            script_address_column = _get_or_create_column(
                worksheet, header_columns, "Script Address"
            )
            county_column = _get_or_create_column(worksheet, header_columns, "County")

        geolocator = Nominatim(
            user_agent="hampton-yards-lease-comps",
            timeout=10,
        )

        geocode = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=1,
            max_retries=1,
            swallow_exceptions=True,
        )

        # -------------------------------------------------------------
        # Fill Latitude and Longitude
        # -------------------------------------------------------------

        address_cache = {}

        for row_number in range(2, worksheet.max_row + 1):
            full_address_cell = worksheet.cell(row=row_number, column=full_address_column)
            latitude_cell = worksheet.cell(row=row_number, column=latitude_column)
            longitude_cell = worksheet.cell(row=row_number, column=longitude_column)

            # Do not modify formulas.
            if "f" in (
                full_address_cell.data_type,
                latitude_cell.data_type,
                longitude_cell.data_type,
            ):
                continue

            # Keep coordinates that are already valid.
            if _valid_coordinates(latitude_cell.value, longitude_cell.value):
                continue

            if _is_blank(full_address_cell.value):
                continue

            full_address = str(full_address_cell.value).strip()

            # Do not search for the same address multiple times.
            if full_address not in address_cache:
                location = geocode(full_address, exactly_one=True)

                address_cache[full_address] = (
                    None
                    if location is None
                    else (round(location.latitude, 6), round(location.longitude, 6))
                )

            coordinates = address_cache[full_address]

            if coordinates is not None:
                latitude_cell.value, longitude_cell.value = coordinates

        # Stop here unless reverse geocoding was requested.
        if not generate_script_address:
            _save(workbook, output_excel_file)
            return

        # -------------------------------------------------------------
        # Fill Script Address and County
        # -------------------------------------------------------------

        reverse = RateLimiter(
            geolocator.reverse,
            min_delay_seconds=1,
            max_retries=1,
            swallow_exceptions=True,
        )

        coordinate_cache = {}

        for row_number in range(2, worksheet.max_row + 1):
            coordinates = _valid_coordinates(
                worksheet.cell(row=row_number, column=latitude_column).value,
                worksheet.cell(row=row_number, column=longitude_column).value,
            )

            if coordinates is None:
                continue

            script_address_cell = worksheet.cell(row=row_number, column=script_address_column)
            county_cell = worksheet.cell(row=row_number, column=county_column)

            needs_script_address = _is_blank(script_address_cell.value)
            needs_county = _is_blank(county_cell.value)

            # Nothing needs to be added to this row.
            if not needs_script_address and not needs_county:
                continue

            coordinate_key = (round(coordinates[0], 6), round(coordinates[1], 6))

            # Do not reverse-geocode the same coordinates twice.
            if coordinate_key not in coordinate_cache:
                location = reverse(coordinate_key, exactly_one=True, language="en")

                coordinate_cache[coordinate_key] = (
                    None if location is None else (location.raw.get("address") or None)
                )

            address = coordinate_cache[coordinate_key]

            if address is None:
                continue

            if needs_script_address:
                script_address = _format_script_address(address)
                if script_address:
                    script_address_cell.value = script_address

            if needs_county:
                county = address.get("county")
                if county:
                    county_cell.value = str(county).strip()

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


# ---------------------------------------------------------------------
# CBSA lookup
# ---------------------------------------------------------------------

def _add_cbsa(input_excel_file,
              sheet_name,
              output_excel_file,
              cbsa_file,
              usecols,
              match_column):
    """
    Fill the CBSA column by matching match_column + State against a
    Census CBSA delineation file. Rows with no match keep their old value.
    """
    import pandas as pd

    # sheet_name=None would make pandas return every sheet, so use the
    # first sheet instead (and the same sheet in openpyxl below).
    pandas_sheet = 0 if sheet_name is None else sheet_name

    df = pd.read_excel(input_excel_file, sheet_name=pandas_sheet)
    df.columns = [str(column).strip() for column in df.columns]
    _require_columns(df.columns, [match_column, "State"])

    # Census file: data starts on row 4, no usable header.
    lookup = pd.read_excel(
        cbsa_file,
        usecols=usecols,
        skiprows=3,
        header=None,
        names=["CBSA", match_column, "State_Code"],
        dtype={"State_Code": str},
    )

    # Pad state codes to 2 digits in case leading zeros were dropped.
    lookup["State_Code"] = lookup["State_Code"].str.strip().str.zfill(2)
    lookup["State"] = lookup["State_Code"].map(STATE_FIPS_TO_ABBR)
    lookup = lookup.dropna(subset=[match_column, "State", "CBSA"])

    # Normalize text on both sides so the match is case/space-insensitive.
    for column in (match_column, "State"):
        lookup[column] = lookup[column].astype("string").str.upper().str.strip()
        df[column] = df[column].astype("string").str.upper().str.strip()

    lookup = lookup.drop_duplicates(subset=[match_column, "State"], keep="first")

    # A left merge keeps df's row order, so rows still line up with the sheet.
    df = df.merge(
        lookup[[match_column, "State", "CBSA"]].rename(columns={"CBSA": "Matched_CBSA"}),
        on=[match_column, "State"],
        how="left",
    )

    if "CBSA" in df.columns:
        df["CBSA"] = df["Matched_CBSA"].combine_first(df["CBSA"])
    else:
        df["CBSA"] = df["Matched_CBSA"]

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name, default_first=True)
        header_columns = _header_columns(worksheet)
        cbsa_column = _get_or_create_column(worksheet, header_columns, "CBSA")

        for row_number, cbsa_value in enumerate(df["CBSA"], start=2):
            worksheet.cell(
                row=row_number,
                column=cbsa_column,
            ).value = None if pd.isna(cbsa_value) else cbsa_value

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


def cbsa_addition_by_city(input_excel_file,
                          sheet_name,
                          output_excel_file,
                          cbsa_file_city):
    """Match on City + State. CBSA file columns: B = CBSA Title, D = City, E = State FIPS."""
    _add_cbsa(input_excel_file, sheet_name, output_excel_file,
              cbsa_file_city, usecols="B,D,E", match_column="City")


def cbsa_addition_by_county(input_excel_file,
                            sheet_name,
                            output_excel_file,
                            cbsa_file_county):
    """Match on County + State. CBSA file columns: D = CBSA Title, H = County, J = State FIPS."""
    _add_cbsa(input_excel_file, sheet_name, output_excel_file,
              cbsa_file_county, usecols="D,H,J", match_column="County")


# ---------------------------------------------------------------------
# Rent, sizes, zoning
# ---------------------------------------------------------------------

def rent_fill(input_excel_file,
              output_excel_file,
              sheet_name=None):
    """
    Fill blank Annual Rent cells with Monthly Rent * 12.

    Good rows are always saved. Any problem rows are reported afterwards.
    """
    workbook = _load_workbook(input_excel_file)
    errors = []

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, ["Monthly Rent"])

        monthly_column = header_columns["Monthly Rent"]
        annual_column = _get_or_create_column(worksheet, header_columns, "Annual Rent")

        for row_number in range(2, worksheet.max_row + 1):
            monthly_cell = worksheet.cell(row=row_number, column=monthly_column)
            annual_cell = worksheet.cell(row=row_number, column=annual_column)

            # Preserve an existing annual rent or formula.
            if not _is_blank(annual_cell.value):
                continue

            # Skip blank monthly rents.
            if _is_blank(monthly_cell.value):
                continue

            # openpyxl cannot calculate formulas.
            if monthly_cell.data_type == "f":
                continue

            monthly_rent = monthly_cell.value

            if not _is_number(monthly_rent):
                errors.append(
                    f"Monthly Rent in row {row_number} is not numeric: {monthly_rent!r}"
                )
                continue

            if monthly_rent <= 0:
                errors.append(
                    f"Monthly Rent in row {row_number} must be greater than 0."
                )
                continue

            annual_cell.value = monthly_rent * 12

        _save(workbook, output_excel_file)

    finally:
        workbook.close()

    if errors:
        raise ValueError(
            "Annual Rent was saved for the valid rows, but these rows were skipped:\n"
            + "\n".join(errors)
        )


def fill_missing_values_for_duplicate_keys(
        input_excel_file,
        output_excel_file,
        fill_columns=None,
        key_columns="Full Address",
        sheet_name=None):
    """For rows sharing the same key, fill blanks from the first row that has a value."""
    fill_columns = [name.strip() for name in _as_list(fill_columns)]
    key_columns = [name.strip() for name in _as_list(key_columns)]

    if not fill_columns:
        raise ValueError("fill_columns must contain at least one column header.")

    if not key_columns:
        raise ValueError("key_columns must contain at least one column header.")

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, [*key_columns, *fill_columns])

        key_column_numbers = [header_columns[name] for name in key_columns]
        fill_column_numbers = [header_columns[name] for name in fill_columns]

        def resolve_style(current_cell, source_cell, column_number):
            """Keep the cell's own style, else the column's, else the source's."""
            if current_cell.has_style:
                return copy(current_cell._style)

            column_dimension = worksheet.column_dimensions.get(
                get_column_letter(column_number)
            )
            if column_dimension is not None and column_dimension.has_style:
                return copy(column_dimension._style)

            return copy(source_cell._style)

        # Pass 1: group every row number under its key value, in sheet order.
        key_rows = {}

        for row_number in range(2, worksheet.max_row + 1):
            key_value = tuple(
                worksheet.cell(row=row_number, column=column_number).value
                for column_number in key_column_numbers
            )

            # A row cannot be matched if any part of its key is blank.
            if any(_is_blank(value) for value in key_value):
                continue

            key_rows.setdefault(key_value, []).append(row_number)

        # Pass 2: for each duplicated key, fill blanks from the first row
        # (in sheet order) that has a value in that column.
        for rows in key_rows.values():
            if len(rows) < 2:
                continue

            for column_number in fill_column_numbers:
                source_cell = next(
                    (
                        worksheet.cell(row=row_number, column=column_number)
                        for row_number in rows
                        if not _is_blank(
                            worksheet.cell(row=row_number, column=column_number).value
                        )
                    ),
                    None,
                )

                # No row sharing this key has a value in this column.
                if source_cell is None:
                    continue

                for row_number in rows:
                    current_cell = worksheet.cell(row=row_number, column=column_number)

                    if _is_blank(current_cell.value):
                        style = resolve_style(current_cell, source_cell, column_number)
                        current_cell.value = source_cell.value
                        current_cell._style = style

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


def cap_sizes_to_property(
    input_excel_file,
    output_excel_file,
    sheet_name=None,
):
    """Cap yard and lease building sizes at their property sizes."""
    column_pairs = [
        ("Yard Size (AC)", "Property (AC)"),
        ("Lease Building (SF)", "Property Building (SF)"),
    ]

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, [name for pair in column_pairs for name in pair])

        for size_column, property_column in column_pairs:
            for row_number in range(2, worksheet.max_row + 1):
                size_cell = worksheet.cell(row=row_number, column=header_columns[size_column])
                property_cell = worksheet.cell(row=row_number, column=header_columns[property_column])

                # Skip comparisons involving formulas.
                if size_cell.data_type == "f" or property_cell.data_type == "f":
                    continue

                # Compare only numeric values.
                if not (_is_number(size_cell.value) and _is_number(property_cell.value)):
                    continue

                # Replace the size only when it exceeds the property size.
                if size_cell.value > property_cell.value:
                    size_cell.value = property_cell.value

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


def adjust_yard_size_for_building(
    input_excel_file, output_excel_file, sheet_name=None
):
    """
    When Yard Size equals Property size (the whole site was counted as yard),
    subtract the building footprint and write it to "Yard Size (AC) Adjusted".

    Run cap_sizes_to_property first.
    """
    inputs = ("Property (AC)", "Yard Size (AC)", "Lease Building (SF)")
    adjusted = "Yard Size (AC) Adjusted"

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, inputs)
        _get_or_create_column(worksheet, header_columns, adjusted)

        for row_number in range(2, worksheet.max_row + 1):
            cells = [
                worksheet.cell(row=row_number, column=header_columns[name])
                for name in (*inputs, adjusted)
            ]

            if any(cell.data_type == "f" for cell in cells):
                continue

            values = [cell.value for cell in cells[:3]]
            if not all(_is_number(value) for value in values):
                continue

            property_size, yard_size, building_sf = values

            # isclose avoids missing matches due to tiny decimal differences.
            if math.isclose(property_size, yard_size, rel_tol=1e-9, abs_tol=1e-6):
                cells[-1].value = yard_size - building_sf / ACRE_SF

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


def standardize_zoning_character(
    input_excel_file, output_excel_file, character="-", sheet_name=None
):
    """
    Within each county, change zoning codes like "I2" to "I-2" when the
    "I-2" spelling also appears in that county.
    """
    if not isinstance(character, str) or not character:
        raise ValueError("character must be a nonempty string.")

    workbook = _load_workbook(input_excel_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, ["County", "Zoning"])

        preferred = {}
        records = []

        # Find preferred zoning spellings within each county.
        for row_number in range(2, worksheet.max_row + 1):
            county_cell = worksheet.cell(row=row_number, column=header_columns["County"])
            zoning_cell = worksheet.cell(row=row_number, column=header_columns["Zoning"])

            if county_cell.data_type == "f" or zoning_cell.data_type == "f":
                continue

            county, zoning = county_cell.value, zoning_cell.value
            if _is_blank(county) or _is_blank(zoning):
                continue

            zoning = str(zoning)
            records.append((county, zoning, zoning_cell))

            if character in zoning:
                preferred.setdefault((county, zoning.replace(character, "")), zoning)

        # Replace matching codes that lack the chosen character.
        for county, zoning, cell in records:
            replacement = preferred.get((county, zoning))
            if character not in zoning and replacement is not None:
                cell.value = replacement

        _save(workbook, output_excel_file)

    finally:
        workbook.close()


# ---------------------------------------------------------------------
# Highway distance and building coverage
# ---------------------------------------------------------------------

def _nearest_highway_miles(highways_latlon, lat, lon, search_degrees):
    """Straight-line miles from (lat, lon) to the nearest highway line."""
    from shapely.geometry import Point

    # Azimuthal equidistant map centered on the property, in meters,
    # so the property sits at (0, 0) and distances from it are accurate.
    local_crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    origin = Point(0, 0)

    # Only reproject highways inside a box around the property (much faster).
    nearby = highways_latlon.cx[
        lon - search_degrees: lon + search_degrees,
        lat - search_degrees: lat + search_degrees,
    ]

    if not nearby.empty:
        miles = nearby.to_crs(local_crs).distance(origin).min() / METERS_PER_MILE

        # Anything outside the box is at least about this far away, so a
        # result inside this radius is guaranteed to be the true nearest.
        safe_miles = 0.95 * search_degrees * 68.7 * math.cos(
            math.radians(min(abs(lat) + search_degrees, 89.0))
        )
        if miles <= safe_miles:
            return float(miles)

    # Nothing close enough in the box: check every highway.
    return float(
        highways_latlon.to_crs(local_crs).distance(origin).min() / METERS_PER_MILE
    )


def add_distance_to_highway(
    input_file, highway_file, latitude_col, longitude_col,
    sheet_name="Sheet1", output_file=None, overwrite=False,
    road_types=("I",), column_name="Distance to Interstate",
    search_degrees=1.0,
):
    """Fill straight-line distance (in MILES) to the nearest highway."""
    import geopandas as gpd
    import pandas as pd

    pandas_sheet = 0 if sheet_name is None else sheet_name

    # Read and validate the coordinates.
    data = pd.read_excel(input_file, sheet_name=pandas_sheet)
    data.columns = [str(column).strip() for column in data.columns]
    latitude = pd.to_numeric(data[latitude_col.strip()], errors="coerce")
    longitude = pd.to_numeric(data[longitude_col.strip()], errors="coerce")
    valid = latitude.between(-90, 90) & longitude.between(-180, 180)

    # Load highway lines.
    highways = gpd.read_file(highway_file)

    # Census route type codes (I = Interstate, U = US route, S = State, ...).
    if "RTTYP" in highways.columns:
        highways = highways.loc[highways["RTTYP"].isin(road_types)]

    highways = highways.loc[
        highways.geom_type.isin(["LineString", "MultiLineString"])
        & ~highways.geometry.is_empty
    ]

    if highways.empty or highways.crs is None:
        raise ValueError("The highway file needs highway lines and a defined CRS.")

    # Lat/lon copy, used to pick out nearby highways quickly.
    highways_latlon = highways.to_crs(epsg=4326)

    workbook = _load_workbook(input_file)

    try:
        worksheet = _get_worksheet(workbook, sheet_name, default_first=True)

        named_headers = [
            str(cell.value).strip() for cell in worksheet[1] if cell.value is not None
        ]
        if len(named_headers) != len(set(named_headers)):
            raise ValueError("The worksheet contains duplicate column headers.")

        header_columns = _header_columns(worksheet)
        column_number = _get_or_create_column(worksheet, header_columns, column_name)

        # Calculate each unique location once.
        distance_cache = {}

        for row_index in data.index[valid]:
            cell = worksheet.cell(row=int(row_index) + 2, column=column_number)

            if not overwrite and not _is_blank(cell.value):
                continue

            location = (float(latitude.at[row_index]), float(longitude.at[row_index]))

            if location not in distance_cache:
                distance_cache[location] = _nearest_highway_miles(
                    highways_latlon, *location, search_degrees
                )

            cell.value = distance_cache[location]

        saved_file = output_file or input_file
        _save(workbook, saved_file)
        return saved_file

    finally:
        workbook.close()


def fill_building_coverage(
    input_excel_file,
    output_excel_file,
    building_sf_column,
    land_ac_column,
    output_column,
    sheet_name=None,
):
    """Fill blank coverage cells using building square feet and land acres."""
    building_sf_column = building_sf_column.strip()
    land_ac_column = land_ac_column.strip()
    output_column = output_column.strip()

    if not output_column:
        raise ValueError("output_column must be a nonempty column header.")

    input_path = Path(input_excel_file)
    output_path = Path(output_excel_file)

    workbook = _load_workbook(input_path)

    try:
        worksheet = _get_worksheet(workbook, sheet_name)
        header_columns = _header_columns(worksheet)
        _require_columns(header_columns, [building_sf_column, land_ac_column])

        changed = output_column not in header_columns
        coverage_column = _get_or_create_column(worksheet, header_columns, output_column)
        building_column = header_columns[building_sf_column]
        land_column = header_columns[land_ac_column]

        for row_number in range(2, worksheet.max_row + 1):
            coverage_cell = worksheet.cell(row=row_number, column=coverage_column)
            if not _is_blank(coverage_cell.value):
                continue

            building_sf = worksheet.cell(row=row_number, column=building_column).value
            land_acres = worksheet.cell(row=row_number, column=land_column).value

            if not (_is_number(building_sf) and _is_number(land_acres)):
                continue

            if building_sf < 0 or land_acres <= 0:
                continue

            coverage_cell.value = building_sf / (land_acres * ACRE_SF)
            if coverage_cell.number_format == "General":
                coverage_cell.number_format = "0.000"
            changed = True

        if changed or input_path.resolve() != output_path.resolve():
            _save(workbook, output_path)

    finally:
        workbook.close()

