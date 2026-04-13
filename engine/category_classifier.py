# engine/category_classifier.py
"""
CategoryAutoClassifier — Veri özelliklerine göre her kategoriyi profil tipine atar.
Grup hiyerarşisi: alt-kategoriler (abiye nikah, abiye düğün) ortak bir
ebeveyn grubundan (abiye) profil parametrelerini miras alır.

Kurallar (öncelik sırasıyla):
  1. CATEGORY_TAG_MAP'te manuel override varsa → onu kullan
  2. KNOWN_GROUPS'ta grup varsa → grubun profil tipini ata
  3. n_products < 5                              → sparse
  4. n_products < 15 AND avg_price > 300         → premium
  5. n_products > 80 OR price_cv > 0.7           → heterogeneous
  6. Geri kalan                                  → mid_fashion

Sonuçlar categories_registry DB tablosuna yazılır (group_name + overrides dahil).
"""
import logging
from collections import defaultdict
from config import (
    CATEGORY_TAG_MAP,
    AUTO_CLASSIFY_RULES,
    DEFAULT_CATEGORY_PROFILE,
)

logger = logging.getLogger(__name__)

# ─── Bilinen Grup Tanımları ─────────────────────────────────────────────────
# Anahtar: grup adı (1 veya 2 kelime)
# Değer: varsayılan profil tipi
# Alt-kategorilerden otomatik tespit + bu dict'teki manuel override
KNOWN_GROUPS = {
    "abiye":       "premium",
    "bikini":      "mid_fashion",
    "blazer":      "mid_fashion",
    "bluz":        "mid_fashion",
    "crop top":    "mid_fashion",
    "elbise":      "mid_fashion",
    "etek":        "mid_fashion",
    "eşofman":     "mid_fashion",
    "gömlek":      "mid_fashion",
    "hırka":       "mid_fashion",
    "jean":        "mid_fashion",
    "kazak":       "mid_fashion",
    "pantolon":    "mid_fashion",
    "pijama":      "mid_fashion",
    "spor":        "mid_fashion",
    "sweatshirt":  "mid_fashion",
    "tayt":        "mid_fashion",
    "tişört":      "mid_fashion",
    "çanta":       "heterogeneous",
    "çorap":       "mid_fashion",
    "şapka":       "mid_fashion",
    "şort":        "mid_fashion",
    "yelek":       "mid_fashion",
}


class CategoryAutoClassifier:
    """Otomatik kategori profil ataması + grup hiyerarşisi."""

    def detect_group(self, category: str) -> str | None:
        """
        Kategori adından grup adını tespit eder.

        Strateji:
          1. İlk 2 kelime KNOWN_GROUPS'ta mı? (crop top baskılı → crop top)
          2. İlk 1 kelime KNOWN_GROUPS'ta mı? (abiye nikah → abiye)
          3. Son kelime KNOWN_GROUPS'ta mı? (kadın abiye → abiye, romantik elbise → elbise)
          4. Bulunamazsa None döndürür (bağımsız kategori)
        """
        words = category.strip().split()
        if len(words) < 2:
            return None

        # 2 kelimelik grup kontrolü (crop top, vb.)
        two_word = f"{words[0]} {words[1]}"
        if two_word in KNOWN_GROUPS:
            if len(words) > 2:
                return two_word
            return None

        # İlk kelime grup kontrolü (abiye nikah → abiye)
        if words[0] in KNOWN_GROUPS:
            return words[0]

        # Son kelime grup kontrolü (kadın abiye → abiye, romantik elbise → elbise)
        if words[-1] in KNOWN_GROUPS:
            return words[-1]

        return None

    def classify_single(self, stats: dict, group_name: str | None = None) -> str:
        """
        Tek bir kategorinin profil tipini belirler.
        Grubu varsa grubun profil tipini alır.
        """
        category = stats.get("category", "")
        rules = AUTO_CLASSIFY_RULES

        # 1. Manuel override
        if category in CATEGORY_TAG_MAP:
            return CATEGORY_TAG_MAP[category]

        # 2. Grup profili
        if group_name and group_name in KNOWN_GROUPS:
            return KNOWN_GROUPS[group_name]

        n_products = stats.get("n_products", 0)
        avg_price  = stats.get("avg_price", 0.0)
        price_cv   = stats.get("price_cv", 0.0)

        # 3-6: Otomatik sınıflandırma
        if n_products < rules["sparse_max_products"]:
            return "sparse"
        if n_products < rules["premium_max_products"] and avg_price > rules["premium_min_avg_price"]:
            return "premium"
        if n_products >= rules["heterogeneous_min_products"] or price_cv > rules["heterogeneous_min_price_cv"]:
            return "heterogeneous"
        return "mid_fashion"

    def _determine_lifecycle(self, n_days: int) -> str:
        if n_days >= 365:
            return "MATURE"
        elif n_days >= 90:
            return "HOT"
        elif n_days >= 30:
            return "WARMING"
        return "COLD"

    def classify_all(self, stats_list: list[dict]) -> dict[str, dict]:
        """
        Tüm kategorileri sınıflandırır, grup ataması yapar ve DB'ye yazar.

        Returns:
            {category: {profile_type, lifecycle, group_name, n_products, n_days}, ...}
        """
        from db.writer import upsert_category_registry

        results = {}
        profile_counts = {}
        group_counts = defaultdict(int)

        # Önce tüm kategorilerin gruplarını tespit et
        for stats in stats_list:
            category = stats["category"]
            group_name = self.detect_group(category)
            profile_type = self.classify_single(stats, group_name)
            lifecycle = self._determine_lifecycle(stats.get("n_days", 0))

            results[category] = {
                "profile_type":   profile_type,
                "lifecycle":      lifecycle,
                "group_name":     group_name,
                "n_products":     stats["n_products"],
                "n_days":         stats.get("n_days", 0),
            }

            # DB'ye kaydet
            upsert_category_registry({
                "search_term":    category,
                "profile_type":   profile_type,
                "lifecycle":      lifecycle,
                "data_days":      stats.get("n_days", 0),
                "total_products": stats["n_products"],
                "feedback_count": 0,
                "kalman_state":   {},
                "group_name":     group_name,
                "overrides":      {},
            })

            profile_counts[profile_type] = profile_counts.get(profile_type, 0) + 1
            if group_name:
                group_counts[group_name] += 1

        # Özet log
        grouped = sum(1 for r in results.values() if r["group_name"])
        standalone = len(results) - grouped
        logger.info(
            f"🏷️  {len(results)} kategori: "
            + ", ".join(f"{k}={v}" for k, v in sorted(profile_counts.items()))
            + f" | {len(group_counts)} grup ({grouped} alt-kategori), "
            + f"{standalone} bağımsız"
        )

        return results
