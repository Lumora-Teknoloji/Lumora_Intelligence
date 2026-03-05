# algorithms/clip_model.py
"""
Algoritma 7: CLIP (OpenAI)
Görsel + metin envanter eşleştirme — zero-shot, eğitim gerektirmez.
"""
import numpy as np
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED


class VisualMatcher:
    """
    CLIP ile ürün fotoğrafı + metin eşleştirme.

    İki temel işlev:
    1. visual_trend_score: Ürün fotoğrafının trend ürünlere benzerliği
    2. text_image_match: "siyah pamuk elbise" yazısı ile fotoğraf eşleştirme

    Zero-shot: eğitim verisi GEREKMİYOR.
    """

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cpu"
        self._loaded = False

    def load(self):
        """CLIP modelini yükler (ilk kullanımda)."""
        if self._loaded:
            return True

        try:
            import torch
            import open_clip

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
            )
            self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"  ✓ CLIP yüklendi ({CLIP_MODEL_NAME}, device={self.device})")
            return True

        except Exception as e:
            print(f"  ⚠ CLIP yüklenemedi: {e}")
            print(f"    → pip install open-clip-torch torch")
            return False

    def encode_image(self, image_path_or_url: str) -> np.ndarray:
        """Fotoğraftan 512 boyutlu embedding çıkarır."""
        if not self.load():
            return np.zeros(512)

        import torch
        from PIL import Image

        try:
            if image_path_or_url.startswith("http"):
                import requests
                from io import BytesIO
                resp = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(resp.content))
            else:
                image = Image.open(image_path_or_url)

            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"  ⚠ Fotoğraf işlenemedi ({image_path_or_url}): {e}")
            return np.zeros(512)

    def encode_text(self, text: str) -> np.ndarray:
        """Metinden 512 boyutlu embedding çıkarır."""
        if not self.load():
            return np.zeros(512)

        import torch

        text_input = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(text_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten()

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """İki embedding arasındaki cosine similarity (0-1)."""
        if np.linalg.norm(embedding1) == 0 or np.linalg.norm(embedding2) == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) /
                     (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def match_text_to_products(self, query: str, product_embeddings: dict) -> list:
        """
        Doğal dil sorgusu ile ürün fotoğraflarını eşleştirir.

        Args:
            query: "siyah pamuk oversize elbise"
            product_embeddings: {product_id: embedding_vector}

        Returns:
            Sıralı [(product_id, similarity_score), ...]
        """
        text_emb = self.encode_text(query)

        scores = []
        for pid, img_emb in product_embeddings.items():
            sim = self.similarity(text_emb, img_emb)
            scores.append({"product_id": pid, "similarity": round(sim, 4)})

        return sorted(scores, key=lambda x: x["similarity"], reverse=True)

    def visual_trend_score(self, product_embedding: np.ndarray,
                           trend_embeddings: list) -> float:
        """
        Bir ürünün trend ürünlere görsel benzerliğini hesaplar.

        Args:
            product_embedding: Ürün fotoğrafı embedding'i
            trend_embeddings: Trend ürünlerin embedding listesi

        Returns:
            0-1 arası trend skoru
        """
        if not trend_embeddings:
            return 0.0

        # Trend ürünlerin ortalama embedding'i
        trend_avg = np.mean(trend_embeddings, axis=0)
        return max(0.0, self.similarity(product_embedding, trend_avg))

    def demo_without_images(self, product_names: list, query: str) -> list:
        """
        Fotoğraf olmadan metin-metin eşleştirme demosu.
        Test için: ürün isimlerini metin olarak encode et ve sorgu ile karşılaştır.
        """
        query_emb = self.encode_text(query)

        results = []
        for name in product_names:
            name_emb = self.encode_text(name)
            sim = self.similarity(query_emb, name_emb)
            results.append({"product_name": name, "similarity": round(sim, 4)})

        return sorted(results, key=lambda x: x["similarity"], reverse=True)
