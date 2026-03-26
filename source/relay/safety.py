import re
from typing import Dict, List, Optional, Tuple

# High-precision regex patterns only.
# Context-aware masking (names, amounts, addresses, etc.) is delegated to the LLM sanitizer.

EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Korean mobile: 010/011/016/017/018/019 + 7~8 digits (with or without dashes)
# US phone: standard NANP format
PHONE_REGEX = (
    r'(?:01[016789]-?\d{3,4}-?\d{4})'           # Korean mobile
    r'|(?:0\d{1,2}-\d{3,4}-\d{4})'              # Korean landline (02-XXXX-XXXX, 031-XXX-XXXX)
    r'|(?:\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b)'  # US
)

# Korean RRN: YYMMDD-[1-4]XXXXXX (with or without dash)
RRN_REGEX = r'\d{6}-[1-4]\d{6}'                 # Korean RRN (with dash)
RRN_NO_DASH_REGEX = r'\b\d{6}[1-4]\d{6}\b'     # Korean RRN (no dash, 13 digits)

SSN_REGEX = r'\b\d{3}-\d{2}-\d{4}\b'            # US SSN
CREDIT_CARD_REGEX = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'

# Korean bank account: 10~14 digits, often with dashes
BANK_ACCOUNT_REGEX = r'\b\d{3,6}-\d{2,6}-\d{4,8}(?:-\d{1,3})?\b'

# Korean business registration number: XXX-XX-XXXXX
BIZ_REG_REGEX = r'\b\d{3}-\d{2}-\d{5}\b'

# Korean passport: M/S/R + 8 digits
PASSPORT_REGEX = r'\b[MSRmsr]\d{8}\b'

# API keys / tokens
AWS_KEY_ID_REGEX = r'\bAKIA[0-9A-Z]{16}\b'
GITHUB_TOKEN_REGEX = r'\b(ghp|gho|ghs|ghr)_[A-Za-z0-9]{36}\b'

# Detect person names in structured "Role: Name" contexts.
# Examples: "Client: Robert Chen", "Patient: Jane Doe", "Employee: Kim Chul-su"
# Matches role keyword (case-insensitive) followed by a Title-Case name (1–4 words).
_ROLE_LABEL_RE = re.compile(
    r'(?i:Client|Patient|Customer|Employee|User|Individual|Claimant|Policyholder|'
    r'Member|Applicant|Subject|Insured|Beneficiary|Contact|Sender|Recipient|'
    r'Payer|Subscriber|Dependent|Guardian|Student|Candidate|Name|Account\s+[Hh]older|'
    r'Policy\s+[Hh]older|Representative|Requestor|Claimant|Appellant)'
    r'(?:\s+\w+)*\s*:\s*'                             # allow "Patient info:" style prefixes
    r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]*\.?){0,3})'  # allow abbreviated last names like "D."
)


class SafetyGuard:
    def __init__(self):
        # High-precision patterns only — order matters (more specific first)
        self.patterns = [
            # Credentials (highest priority — most specific patterns first)
            ("AWS_KEY_ID",     re.compile(AWS_KEY_ID_REGEX)),
            ("GITHUB_TOKEN",   re.compile(GITHUB_TOKEN_REGEX)),
            # Financial
            ("CARD_NUMBER",    re.compile(CREDIT_CARD_REGEX)),
            ("ACCOUNT_ID",     re.compile(BANK_ACCOUNT_REGEX)),
            # Identity
            ("ID",             re.compile(RRN_REGEX)),          # Korean RRN with dash
            ("ID",             re.compile(RRN_NO_DASH_REGEX)),  # Korean RRN no dash
            ("ID",             re.compile(SSN_REGEX)),           # US SSN
            ("ID",             re.compile(PASSPORT_REGEX)),      # Korean passport
            ("BIZ_REG",        re.compile(BIZ_REG_REGEX)),      # Korean biz reg no.
            # Contact
            ("EMAIL",          re.compile(EMAIL_REGEX)),
            ("PHONE",          re.compile(PHONE_REGEX)),
        ]

    def scan_and_redact(self, text: str) -> Tuple[str, bool, List[str], Dict[str, str]]:
        """
        Scans text for PII and builds a reversible binding map.
        Returns: (redacted_text, pii_detected, list_of_notes, binding_map)
        binding_map: {placeholder -> original_value}, e.g. {"[ID]": "800101-1234567"}
        Multiple occurrences of the same type use [LABEL_2], [LABEL_3], etc.
        """
        binding_map: Dict[str, str] = {}
        redacted_text = text
        notes = []
        pii_found = False

        for label, pattern in self.patterns:
            counter = [0]

            def make_replacer(lbl, ctr, bmap):
                def replacer(m):
                    val = m.group()
                    ctr[0] += 1
                    key = f"[{lbl}]" if ctr[0] == 1 else f"[{lbl}_{ctr[0]}]"
                    bmap[key] = val
                    return key
                return replacer

            new_text = pattern.sub(make_replacer(label, counter, binding_map), redacted_text)
            if counter[0] > 0:
                pii_found = True
                notes.append(f"Detected {counter[0]} potential {label}(s).")
            redacted_text = new_text

        return redacted_text, pii_found, notes, binding_map

    def scan_names_in_context(self, text: str, binding_map: Dict[str, str]) -> str:
        """
        Detect person names that appear after role-labelled keys such as
        "Client: Robert Chen" or "Patient: Jane Doe".  Updates *binding_map*
        in-place and returns the redacted text.

        Two passes are applied:
          1. Regex finds "Role: Name" patterns → assigns [NAME] / [NAME_N] keys.
          2. All remaining occurrences of each discovered name are replaced so
             the name cannot appear elsewhere in the text.
        """
        # Build reverse lookup for names already captured (shouldn't happen in
        # practice because Stage 1 patterns don't produce [NAME], but be safe).
        value_to_key: Dict[str, str] = {
            v: k for k, v in binding_map.items() if k.startswith('[NAME')
        }
        name_count = sum(1 for k in binding_map if k.startswith('[NAME'))

        def _replace(m: re.Match) -> str:
            nonlocal name_count
            name = m.group(1)
            if name in value_to_key:
                key = value_to_key[name]
            else:
                name_count += 1
                key = '[NAME]' if name_count == 1 else f'[NAME_{name_count}]'
                binding_map[key] = name
                value_to_key[name] = key
            # Replace only the name part; keep the "Role: " prefix intact.
            full_match = m.group(0)
            return full_match.replace(name, key, 1)

        # Pass 1: detect names in structured context labels
        text = _ROLE_LABEL_RE.sub(_replace, text)

        # Pass 2: replace all remaining occurrences of each discovered name
        for key, name in list(binding_map.items()):
            if key.startswith('[NAME') and name in text:
                text = text.replace(name, key)

        return text

    def check_leakage(
        self,
        original: str,
        sanitized: str,
        binding_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Check if sanitized text still contains PII from the original.
        Checks both regex-detectable patterns and any values in *binding_map*.
        Returns: (leakage_found, list_of_leaked_items)
        """
        leaked = []
        for label, pattern in self.patterns:
            for match in pattern.finditer(original):
                pii_value = match.group()
                if pii_value in sanitized:
                    leaked.append(f"{label}:{pii_value[:4]}***")
        # Also verify that binding_map values (e.g. names found contextually)
        # did not survive sanitization.
        if binding_map:
            for key, value in binding_map.items():
                if len(value) > 3 and value in sanitized:
                    label = key.strip('[]').split('_')[0]
                    leaked.append(f"{label}:{value[:4]}***")
        return len(leaked) > 0, leaked
