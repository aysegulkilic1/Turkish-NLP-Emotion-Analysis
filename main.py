import pandas as pd
import string
from transformers import pipeline
import warnings


# 1. Veri seti simülasyonu:

print("Veri Seti Yükleniyor...")

gelen_veri = [
    {"id": 1, "yorum": "Ürün harika paketlenmişti ama kargo çok geç geldi!!"},
    {"id": 2, "yorum": "BERBAT BİR ÜRÜN, SAKIN ALMAYIN... param çöpe gitti."},
    {"id": 3, "yorum": "Fiyat performans ürünü, gayet başarılı teşekkürler."},
    {"id": 4, "yorum": "Kutusu ezilmişti, iade etmek zorunda kaldım."},
    {"id": 5, "yorum": "Müşteri hizmetleri çok ilgiliydi, sorunu hemen çözdüler."}
]

# Veriyi pandas DataFrame'e çeviriyoruz:
df = pd.DataFrame(gelen_veri)
print(f"Toplam {len(df)} adet yorum işleme alındı.\n")


# 2. Ön işleme fonksiyonu (Preprocessing):

def metin_temizle(gelen_metin):

    # Normalization (Küçük harfe çevirir):
    temizlenen = gelen_metin.lower()

    # Punctuation Removal (Noktalama işaretlerini temizler):
    for isaret in string.punctuation:
        temizlenen = temizlenen.replace(isaret, "")

    # Trimming (Boşlukları temizler):
    temizlenen = temizlenen.strip()

    return temizlenen


# Temizlik işlemini tüm veri setine uyguluyoruz:
print("Metinler Temizleniyor (Preprocessing)...")
df["temiz_yorum"] = df["yorum"].apply(metin_temizle)
print("Temizlik tamamlandı.\n")


# 3. Yapay zeka modelini yüklüyoruz:
print("Yapay Zeka Modeli Yükleniyor (Hugging Face)...")

analizci = pipeline("sentiment-analysis",
                    model="savasy/bert-base-turkish-sentiment-cased",
                    framework="pt")
print("Model başarıyla yüklendi\n")


# 4. Analiz Döngüsü:
print("Analiz Başladı (AI + Rule Based)...")
sonuclar_sepeti = []

for yorum in df["temiz_yorum"]:

    # A) Yapay zeka ile duygu analizi (Sentiment Analysis) yapıyoruz:
    ai_cevabi = analizci(yorum)[0]
    etiket = ai_cevabi['label']
    skor = ai_cevabi['score']

    # Etiket standardizasyonu yapıyoruz:
    if etiket == "positive" or etiket == "LABEL_1":
        duygu_durumu = "OLUMLU"
    else:
        duygu_durumu = "OLUMSUZ"

    # B) Kural tabanlı kategori analizi (Keyword Extraction) yapıyoruz:
    kategori = "Genel"
    if "kargo" in yorum or "paket" in yorum or "teslimat" in yorum:
        kategori = "Lojistik / Kargo"
    elif "hizmet" in yorum or "iade" in yorum or "ilgili" in yorum:
        kategori = "Müşteri Hizmetleri"
    elif "fiyat" in yorum or "performans" in yorum:
        kategori = "Fiyat/Performans"

    # Sonuçları kaydet:
    sonuclar_sepeti.append({
        "Sentiment (Duygu)": duygu_durumu,
        "Güven Skoru": round(skor, 2),
        "Kategori": kategori
    })

# 4. RAPORLAMA:
# Analiz sonuçlarını ana tablo ile birleştiriyoruz:
sonuc_df = pd.DataFrame(sonuclar_sepeti)
final_tablo = pd.concat([df, sonuc_df], axis=1)

print("\n" + "=" * 50)
print("SONUÇ RAPORU")
print("=" * 50)

pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.width', 1000)  # Genişlik ayarı

print(final_tablo[["yorum", "Sentiment (Duygu)", "Güven Skoru", "Kategori"]])
print("\nİşlem Başarıyla Tamamlandı.")

