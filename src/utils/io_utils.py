import json


def load_json_with_fallback(path, encodings_to_try=None):
    if encodings_to_try is None:
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    last_error = None

    for encoding in encodings_to_try:
        try:
            with open(path, "r", encoding=encoding) as f:
                data = json.load(f)
            print(f"Loaded JSON from {path} with encoding: {encoding}")
            return data
        except UnicodeDecodeError as e:
            last_error = e

    if last_error is not None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            (
                f"Failed to decode JSON file '{path}' with supported encodings: "
                f"{', '.join(encodings_to_try)}"
            ),
        )

    raise ValueError(f"Failed to decode JSON file '{path}'")
