import json
import re
from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------

OCR_PATH = Path(
    "results/paddle_ocr/structured_ocr.json"
)

OUTPUT_DIR = Path(
    "results/field_extraction"
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "extracted_fields.json"
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def clean_text(text):

    return " ".join(
        str(text).strip().split()
    )


def valid_confidence(item, threshold=0.50):

    confidence = item.get(
        "confidence"
    )

    if confidence is None:
        return True

    return confidence >= threshold


# -------------------------------------------------
# Main extraction
# -------------------------------------------------

def main():

    print("=" * 60)
    print("STRUCTURED FIELD EXTRACTION")
    print("=" * 60)

    if not OCR_PATH.exists():

        raise FileNotFoundError(
            f"OCR file not found:\n{OCR_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------
    # Load OCR
    # ---------------------------------------------

    with open(
        OCR_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        ocr_data = json.load(
            file
        )

    # Keep reasonably confident OCR detections

    items = []

    for item in ocr_data:

        if not valid_confidence(
            item,
            threshold=0.50
        ):
            continue

        text = clean_text(
            item.get(
                "text",
                ""
            )
        )

        if text:

            items.append({
                "text": text,
                "confidence": item.get(
                    "confidence"
                )
            })

    texts = [
        item["text"]
        for item in items
    ]

    print(
        f"\nUsable OCR detections: "
        f"{len(texts)}"
    )

    # ---------------------------------------------
    # Initialize fields
    # ---------------------------------------------

    fields = {

        "surname":
            None,

        "given_name":
            None,

        "nationality":
            None,

        "document_number":
            None,

        "date_of_birth":
            None,

        "sex":
            None,

        "date_of_issue":
            None,

        "date_of_expiry":
            None,

        "personal_number":
            None
    }

    # ---------------------------------------------
    # Extract dates
    # ---------------------------------------------

    date_pattern = re.compile(
        r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b"
    )

    dates = []

    for text in texts:

        matches = date_pattern.findall(
            text
        )

        dates.extend(
            matches
        )

    # Normalize separators

    dates = [
        date.replace(
            "/",
            "-"
        ).replace(
            ".",
            "-"
        )
        for date in dates
    ]

    print(
        "\nDates detected:",
        dates
    )

    # For this known document layout/order:
    #
    # DOB -> Issue -> Expiry

    if len(dates) >= 1:

        fields[
            "date_of_birth"
        ] = dates[0]

    if len(dates) >= 2:

        fields[
            "date_of_issue"
        ] = dates[1]

    if len(dates) >= 3:

        fields[
            "date_of_expiry"
        ] = dates[2]

    # ---------------------------------------------
    # Extract numeric IDs
    # ---------------------------------------------

    numeric_candidates = []

    for text in texts:

        cleaned = re.sub(
            r"\s+",
            "",
            text
        )

        if re.fullmatch(
            r"\d{9,10}",
            cleaned
        ):

            numeric_candidates.append(
                cleaned
            )

    print(
        "Numeric candidates:",
        numeric_candidates
    )

    # Current Albanian document sample:
    #
    # 9-digit value -> document/card number
    # 10-digit value -> personal number

    for number in numeric_candidates:

        if (
            len(number) == 9
            and
            fields["document_number"]
            is None
        ):

            fields[
                "document_number"
            ] = number

        elif (
            len(number) == 10
            and
            fields["personal_number"]
            is None
        ):

            fields[
                "personal_number"
            ] = number

    # ---------------------------------------------
    # Sex
    # ---------------------------------------------

    for text in texts:

        if text.upper() in {
            "M",
            "F"
        }:

            fields[
                "sex"
            ] = text.upper()

            break

    # ---------------------------------------------
    # Nationality
    # ---------------------------------------------

    for text in texts:

        lower = text.lower()

        if (
            "alban" in lower
            or
            "shqipt" in lower
        ):

            fields[
                "nationality"
            ] = "Albanian"

            break

    # ---------------------------------------------
    # Name extraction
    #
    # Initial layout-specific rules.
    # Later we will replace these with
    # bounding-box/layout-based extraction.
    # ---------------------------------------------

    ignored_words = {

        "F",
        "NB"
    }

    name_candidates = []

    for item in items:

        text = item[
            "text"
        ]

        confidence = item[
            "confidence"
        ]

        # Must be alphabetic-ish

        if not re.fullmatch(
            r"[A-Za-zÀ-ÿ]+",
            text
        ):

            continue

        if text.upper() in ignored_words:

            continue

        # Avoid obvious document labels

        lower = text.lower()

        blocked = [
            "republic",
            "shqip",
            "alban",
            "signature",
            "authority"
        ]

        if any(
            word in lower
            for word in blocked
        ):

            continue

        # Names should have strong OCR confidence

        if (
            confidence is not None
            and
            confidence < 0.90
        ):

            continue

        name_candidates.append(
            text
        )

    print(
        "Possible names:",
        name_candidates
    )

    if len(name_candidates) >= 1:

        fields[
            "surname"
        ] = name_candidates[0]

    if len(name_candidates) >= 2:

        fields[
            "given_name"
        ] = name_candidates[1]

    # ---------------------------------------------
    # Print result
    # ---------------------------------------------

    print(
        "\n" +
        "=" * 60
    )

    print(
        "EXTRACTED DOCUMENT DATA"
    )

    print(
        "=" * 60
    )

    for key, value in fields.items():

        print(
            f"{key:20}: "
            f"{value}"
        )

    # ---------------------------------------------
    # Save JSON
    # ---------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            fields,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()