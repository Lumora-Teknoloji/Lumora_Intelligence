# engine/slug.py
"""
Kategori adlarını dosya sistemi ve model dizini için güvenli slug'lara dönüştürür.
Türkçe → ASCII, stop word çıkarma, boşluk → alt çizgi.

Örnekler:
    "abiye nikah"     → "abiye_nikah"
    "şort denim"      → "sort_denim"
    "çanta omuz"      → "canta_omuz"
    "eşofman takım"   → "esofman_takim"
    "crop top baskılı" → "crop_top_baskili"
"""
import re
import unicodedata

# Türkçe → ASCII haritalama (özel karakterler)
_TR_MAP = str.maketrans({
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S",
    "ü": "u", "Ü": "U",
})

# Türkçe stop words (katma değeri düşük, opsiyonel)
_STOP_WORDS = {"kadın", "erkek", "kadin", "erkek", "için", "icin", "ve", "ile"}


def slugify(text: str) -> str:
    """
    Kategori adını dosya sistemi güvenli slug'a dönüştürür.

    1. Küçük harfe çevir
    2. Türkçe özel karakterleri ASCII'ye çevir
    3. Stop word'leri çıkar
    4. Alfanumerik ve boşluk dışındakileri kaldır
    5. Boşlukları alt çizgiye çevir
    """
    if not text:
        return "unknown"

    s = text.lower().strip()
    s = s.translate(_TR_MAP)

    # Kalan unicode → ASCII (ek güvenlik)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    # Stop words çıkar
    words = s.split()
    words = [w for w in words if w not in _STOP_WORDS]

    # Non-alfanumerik kaldır
    s = " ".join(words)
    s = re.sub(r"[^a-z0-9 ]", "", s)

    # Boşluk → underscore
    s = re.sub(r"\s+", "_", s).strip("_")

    return s or "unknown"
