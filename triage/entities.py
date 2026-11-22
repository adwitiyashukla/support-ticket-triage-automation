import re

CURRENCY_WORDS = {
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "usd": "USD", "dollar": "USD", "dollars": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "inr": "INR", "rupee": "INR", "rupees": "INR",
    "chf": "CHF", "sek": "SEK", "pln": "PLN", "aud": "AUD", "cad": "CAD", "jpy": "JPY",
}
CURRENCY_SYMBOLS = {"$": "USD", "\u00a3": "GBP", "\u20ac": "EUR", "\u20b9": "INR", "\u00a5": "JPY"}

AMOUNT = re.compile("(?P<symbol>[$\u00a3\u20ac\u20b9\u00a5])?\\s?(?P<value>\\d{1,3}(?:,\\d{3})*(?:\\.\\d{1,2})?|\\d+(?:\\.\\d{1,2})?)")
LAST4 = re.compile(r"(?:ending(?:\s+in)?|ends\s+with|last\s+four|last\s+4|card\s+number)\D{0,12}(\d{4})\b", re.IGNORECASE)
MASKED = re.compile(r"(?:\*{2,}|x{2,})\s*(\d{4})\b", re.IGNORECASE)
NUMERIC_DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2})\b")
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
TEXT_DATE = re.compile(rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})|(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?)\b", re.IGNORECASE)
MERCHANT = re.compile(r"\b(?:at|from|to|with)\s+((?:[A-Z][\w&'.-]*)(?:\s+[A-Z][\w&'.-]*){0,2})")
MERCHANT_STOPWORDS = {"I", "My", "The", "A", "An", "It", "You", "We", "This", "That", "ATM", "PIN", "UK", "US"}
CURRENCY_TOKEN = re.compile(r"\b([a-z]{3,7})\b", re.IGNORECASE)


def _amount(text):
    best_value = ""
    best_currency = ""
    for match in AMOUNT.finditer(text):
        raw = match.group("value")
        if raw.isdigit() and len(raw) == 4 and int(raw) > 1900:
            continue
        if raw.isdigit() and len(raw) > 4:
            continue
        value = raw.replace(",", "")
        window = text[max(0, match.start() - 12):match.end() + 12]
        currency = ""
        if match.group("symbol"):
            currency = CURRENCY_SYMBOLS.get(match.group("symbol"), "")
        if not currency:
            for token in CURRENCY_TOKEN.findall(window):
                if token.lower() in CURRENCY_WORDS:
                    currency = CURRENCY_WORDS[token.lower()]
                    break
        if not best_value or (currency and not best_currency):
            best_value, best_currency = value, currency
        if best_value and best_currency:
            break
    return best_value, best_currency


def _currency_only(text):
    for token in CURRENCY_TOKEN.findall(text):
        if token.lower() in CURRENCY_WORDS:
            return CURRENCY_WORDS[token.lower()]
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return ""


def _card_last4(text):
    match = LAST4.search(text) or MASKED.search(text)
    return match.group(1) if match else ""


def _date(text):
    match = NUMERIC_DATE.search(text)
    if match:
        return match.group(1)
    match = TEXT_DATE.search(text)
    return match.group(1) if match else ""


def _merchant(text):
    for match in MERCHANT.finditer(text):
        candidate = match.group(1).strip()
        head = candidate.split()[0]
        if head in MERCHANT_STOPWORDS:
            continue
        if len(candidate) < 3:
            continue
        return candidate
    return ""


def extract(text, types):
    amount, currency = _amount(text)
    if not currency:
        currency = _currency_only(text)
    found = {
        "amount": amount,
        "currency": currency,
        "card_last4": _card_last4(text),
        "date": _date(text),
        "merchant": _merchant(text),
    }
    return {key: found.get(key, "") for key in types}
