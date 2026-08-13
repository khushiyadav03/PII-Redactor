"""
Hinglish: Ye module original PII value ko ek CONSISTENT fake value se map
karta hai. Agar "Rahul Sharma" document mein 5 baar aata hai, to paanchon
jagah wahi fake naam ("Alex Morgan" jaisa kuch) use hona chahiye - alag
alag nahi. Isliye ek dict-based cache (original -> fake) rakhte hain.

Determinism ke liye Faker ko seeded random se drive karte hain (based on
hash of original value) - taaki same input document par re-run karne se
same fake output mile (reproducible testing ke liye zaroori).
"""
import hashlib
import re
from typing import Dict

from faker import Faker

_faker = Faker()
_faker.seed_instance(42)  # base seed; per-value determinism neeche seeded rng se aati hai


class ConsistentReplacer:
    """
    Hinglish: original_value -> fake_value ka mapping cache karta hai,
    per PII-type (taaki "Sharma" ko kabhi name field mein aur kabhi
    company field mein use hone par confuse na ho).
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, str]] = {}

    def _seeded_faker(self, category: str, original: str) -> Faker:
        """Hinglish: original value ke hash se ek deterministic seed banate hain."""
        seed_str = f"{category}:{original.strip().lower()}"
        seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
        f = Faker()
        f.seed_instance(seed_int)
        return f

    def _get_or_create(self, category: str, original: str, generator_fn) -> str:
        bucket = self._cache.setdefault(category, {})
        key = original.strip()
        if key in bucket:
            return bucket[key]
        fake_value = generator_fn(self._seeded_faker(category, original))
        bucket[key] = fake_value
        return fake_value

    def fake_name(self, original: str) -> str:
        def gen(f):
            name = f.name()
            # Hinglish: agar fake naam original se bahut zyada lamba hai to
            # table layout todh sakta hai - chhota alternative try karte hain.
            if len(name) > len(original) * 1.6 + 4:
                name = f.first_name() + " " + f.last_name()[:1] + "."
            return name
        return self._get_or_create("name", original, gen)

    def fake_email(self, original: str) -> str:
        return self._get_or_create("email", original, lambda f: f.email())

    def fake_phone(self, original: str) -> str:
        def gen(f):
            # Hinglish: original format (+91 ya nahi) ko roughly preserve
            # karte hain taaki document ka visual layout na bigde.
            digits_only = re.sub(r"\D", "", original)
            fake_digits = "".join(f.random.choices("0123456789", k=len(digits_only)))
            # Hinglish: pehla digit 0/1 na ho (Indian mobile 6-9 se start
            # hota hai) - realistic dikhne ke liye.
            fake_digits = str(f.random.randint(6, 9)) + fake_digits[1:]
            result = original
            di = iter(fake_digits)
            return re.sub(r"\d", lambda _: next(di), original)
        return self._get_or_create("phone", original, gen)

    def fake_company(self, original: str) -> str:
        return self._get_or_create("company", original, lambda f: f.company())

    def fake_address(self, original: str) -> str:
        return self._get_or_create("address", original, lambda f: f.address().replace("\n", ", "))

    def fake_ssn(self, original: str) -> str:
        return self._get_or_create("ssn", original, lambda f: f.ssn())

    def fake_credit_card(self, original: str) -> str:
        return self._get_or_create("credit_card", original, lambda f: f.credit_card_number())

    def fake_dob(self, original: str) -> str:
        def gen(f):
            d = f.date_of_birth(minimum_age=20, maximum_age=60)
            # Hinglish: original date format (DD/MM/YYYY vs "December 10, 2025")
            # jitna ho sake preserve karte hain.
            if re.search(r"[A-Za-z]", original):
                return d.strftime("%B %d, %Y")
            if original.count("/") == 2:
                return d.strftime("%d/%m/%Y")
            return d.strftime("%d-%m-%Y")
        return self._get_or_create("dob", original, gen)

    def fake_ip(self, original: str) -> str:
        return self._get_or_create("ip", original, lambda f: f.ipv4_public())

    def fake_pan(self, original: str) -> str:
        def gen(f):
            letters = "".join(f.random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
            digits = "".join(f.random.choices("0123456789", k=4))
            last = f.random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            return f"{letters}{digits}{last}"
        return self._get_or_create("pan", original, gen)

    def fake_aadhaar(self, original: str) -> str:
        def gen(f):
            digits = "".join(f.random.choices("0123456789", k=12))
            return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"
        return self._get_or_create("aadhaar", original, gen)

    def fake_passport(self, original: str) -> str:
        def gen(f):
            letter = f.random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
            digits = "".join(f.random.choices("0123456789", k=7))
            return f"{letter}{digits}"
        return self._get_or_create("passport", original, gen)
