# Lumora Intelligence — VPS Deployment Guide

## Gereksinimler
- Ubuntu 20.04+ / Debian 11+
- Python 3.10+
- PostgreSQL 14+ (backend ile aynı sunucu veya bağlantı)
- Nginx
- Certbot (HTTPS için)

---

## Adım 1 — Kodu Çek

```bash
cd /home/ubuntu
git clone https://github.com/yourorg/LancChain_Intelligence.git
cd LancChain_Intelligence
```

## Adım 2 — Python Ortamı

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Adım 3 — .env Dosyası Oluştur

```bash
cp .env.example .env
nano .env
```

**Zorunlu değişkenler:**
```
APP_ENV=production
POSTGRESQL_HOST=localhost
POSTGRESQL_DATABASE=lumora_db
POSTGRESQL_USERNAME=postgres
POSTGRESQL_PASSWORD=güçlü-şifre
INTERNAL_API_KEY=backend-ile-aynı-key
BACKEND_URL=http://localhost:8000
BACKEND_CALLBACK_URL=http://localhost:8000/api/intelligence/callback
```

## Adım 4 — systemd Service

```bash
sudo cp deploy/lumora-intelligence.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lumora-intelligence
sudo systemctl start lumora-intelligence
sudo systemctl status lumora-intelligence
```

## Adım 5 — Nginx Yapılandırması

```bash
# Domain'i güncelle:
sudo nano deploy/nginx-intelligence.conf
# intelligence.yourdomain.com → gerçek domainin ile değiştir

sudo cp deploy/nginx-intelligence.conf /etc/nginx/sites-available/lumora-intelligence
sudo ln -s /etc/nginx/sites-available/lumora-intelligence /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Adım 6 — HTTPS (Let's Encrypt)

```bash
sudo certbot --nginx -d intelligence.yourdomain.com
```

---

## Doğrulama

```bash
# Servis çalışıyor mu?
sudo systemctl status lumora-intelligence

# Health check:
curl http://localhost:8001/health

# Loglar:
sudo journalctl -u lumora-intelligence -f
```

---

## Güncelleme (Yeni Versiyon)

```bash
cd /home/ubuntu/LancChain_Intelligence
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart lumora-intelligence
```

---

## VPS ↔ Backend Ağ Konfigürasyonu

> Intelligence ve Backend **aynı sunucuda** çalışıyorsa hiçbir şey değiştirme.

> **Ayrı sunucularda** ise:
> ```
> # Intelligence .env → backend'in IP'sini ver:
> BACKEND_URL=http://10.0.0.2:8000
> BACKEND_CALLBACK_URL=http://10.0.0.2:8000/api/intelligence/callback
>
> # Backend .env → Intelligence'ın IP'sini ver:
> INTELLIGENCE_URL=http://10.0.0.3:8001
> ```
