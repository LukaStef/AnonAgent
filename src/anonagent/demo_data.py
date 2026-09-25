"""Sample documents for the demo.

All values here are invented. They exist to show the guard working on the
three cases an audience recognizes instantly: a bank, a hospital, a codebase.
"""

from __future__ import annotations

SAMPLES: dict[str, str] = {
    # The balance has to be here. Ask whether someone can afford a transfer
    # without saying what they have, and the model rightly answers that it
    # cannot tell -- which looks like the masking failed when it did not.
    "banking": (
        "James Bond, account holder at our Belgrade branch, has a balance of "
        "62,400 RSD and requested a transfer of 50,000 RSD. His IBAN is "
        "RS35260005601001611379 and his registered email is "
        "james.bond@example.com. The request came from employee EMP-004217. "
        "Does he have sufficient funds, and is anything about this unusual?"
    ),
    "medical": (
        "Patient Maria Whitfield, SSN 432-56-7890, was admitted on the 14th with "
        "chest pain and shortness of breath. Contact number is +1 415 555 0182. "
        "Summarize the likely differential diagnosis and the next tests to order."
    ),
    "code": (
        "Review this snippet for security problems:\n\n"
        "    client = OpenAI(api_key='sk-proj-4Xq82LmZpR7vNw1KdTfA9BcE')\n"
        "    DB_HOST = '10.0.14.201'\n"
        "    ADMIN = 'james.calloway@internal.example.com'\n\n"
        "Tell me what an attacker could do with what is exposed here."
    ),
}

DEFAULT_SAMPLE = "banking"
