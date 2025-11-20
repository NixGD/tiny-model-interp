"""ASCII normalization for converting non-ASCII characters to ASCII equivalents."""

# Character normalization map: non-ASCII -> ASCII equivalents
# This preserves semantic meaning while converting to ASCII
CHAR_NORMALIZATION_MAP = {
    # Smart quotes
    "\u201c": '"',  # Left double quotation mark
    "\u201d": '"',  # Right double quotation mark
    "\u2018": "'",  # Left single quotation mark
    "\u2019": "'",  # Right single quotation mark
    # Dashes
    "\u2013": "-",  # En dash
    "\u2014": "-",  # Em dash
    # Ellipsis
    "\u2026": "...",  # Horizontal ellipsis
    # Other common punctuation
    "„": '"',  # U+201E Double low-9 quotation mark
    "‚": "'",  # U+201A Single low-9 quotation mark
    "‹": "<",  # U+2039 Single left-pointing angle quotation mark
    "›": ">",  # U+203A Single right-pointing angle quotation mark
    "«": '"',  # U+00AB Left-pointing double angle quotation mark
    "»": '"',  # U+00BB Right-pointing double angle quotation mark
    # Spaces
    "\xa0": " ",  # U+00A0 Non-breaking space
    "\u2000": " ",  # En quad
    "\u2001": " ",  # Em quad
    "\u2002": " ",  # En space
    "\u2003": " ",  # Em space
    "\u2004": " ",  # Three-per-em space
    "\u2005": " ",  # Four-per-em space
    "\u2006": " ",  # Six-per-em space
    "\u2007": " ",  # Figure space
    "\u2008": " ",  # Punctuation space
    "\u2009": " ",  # Thin space
    "\u200a": " ",  # Hair space
    "\u202f": " ",  # Narrow no-break space
    "\u205f": " ",  # Medium mathematical space
    # Hyphens and dashes
    "\u2010": "-",  # Hyphen
    "\u2011": "-",  # Non-breaking hyphen
    "\u2012": "-",  # Figure dash
    "\u2015": "-",  # Horizontal bar
    # Other common characters
    "°": " degrees ",  # U+00B0 Degree sign (convert to words for clarity)
    "×": "x",  # U+00D7 Multiplication sign
    "÷": "/",  # U+00F7 Division sign
    "€": "EUR",  # U+20AC Euro sign
    "£": "GBP",  # U+00A3 Pound sign
    "¥": "JPY",  # U+00A5 Yen sign
    "©": "(c)",  # U+00A9 Copyright sign
    "®": "(R)",  # U+00AE Registered sign
    "™": "(TM)",  # U+2122 Trade mark sign
    # Common accented vowels and characters (Latin-1 Supplement)
    # Lowercase
    "á": "a",  # U+00E1
    "à": "a",  # U+00E0
    "â": "a",  # U+00E2
    "ã": "a",  # U+00E3
    "ä": "a",  # U+00E4
    "å": "a",  # U+00E5
    "æ": "ae",  # U+00E6
    "ç": "c",  # U+00E7
    "è": "e",  # U+00E8
    "é": "e",  # U+00E9
    "ê": "e",  # U+00EA
    "ë": "e",  # U+00EB
    "ì": "i",  # U+00EC
    "í": "i",  # U+00ED
    "î": "i",  # U+00EE
    "ï": "i",  # U+00EF
    "ñ": "n",  # U+00F1
    "ò": "o",  # U+00F2
    "ó": "o",  # U+00F3
    "ô": "o",  # U+00F4
    "õ": "o",  # U+00F5
    "ö": "o",  # U+00F6
    "ø": "o",  # U+00F8
    "ù": "u",  # U+00F9
    "ú": "u",  # U+00FA
    "û": "u",  # U+00FB
    "ü": "u",  # U+00FC
    "ý": "y",  # U+00FD
    "ÿ": "y",  # U+00FF
    # Uppercase
    "Á": "A",  # U+00C1
    "À": "A",  # U+00C0
    "Â": "A",  # U+00C2
    "Ã": "A",  # U+00C3
    "Ä": "A",  # U+00C4
    "Å": "A",  # U+00C5
    "Æ": "AE",  # U+00C6
    "Ç": "C",  # U+00C7
    "È": "E",  # U+00C8
    "É": "E",  # U+00C9
    "Ê": "E",  # U+00CA
    "Ë": "E",  # U+00CB
    "Ì": "I",  # U+00CC
    "Í": "I",  # U+00CD
    "Î": "I",  # U+00CE
    "Ï": "I",  # U+00CF
    "Ñ": "N",  # U+00D1
    "Ò": "O",  # U+00D2
    "Ó": "O",  # U+00D3
    "Ô": "O",  # U+00D4
    "Õ": "O",  # U+00D5
    "Ö": "O",  # U+00D6
    "Ø": "O",  # U+00D8
    "Ù": "U",  # U+00D9
    "Ú": "U",  # U+00DA
    "Û": "U",  # U+00DB
    "Ü": "U",  # U+00DC
    "Ý": "Y",  # U+00DD
    "Ð": "D",  # U+00D0 (Icelandic)
    "Þ": "Th",  # U+00DE (Icelandic)
    "ß": "ss",  # U+00DF (German)
}


def normalize_text(text: str) -> str:
    """Normalize text by converting non-ASCII characters to ASCII equivalents.

    Args:
        text: Input text that may contain non-ASCII characters

    Returns:
        Normalized text with non-ASCII characters converted to ASCII equivalents
    """
    result = []
    for char in text:
        # Check if character is in our normalization map
        if char in CHAR_NORMALIZATION_MAP:
            replacement = CHAR_NORMALIZATION_MAP[char]
            # Handle both single and multi-character replacements
            result.extend(replacement)
        elif ord(char) < 128:
            # Already ASCII, keep as is
            result.append(char)
        # Otherwise, skip non-ASCII characters not in our map
    return "".join(result)
