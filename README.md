# Türkçe E-Ticaret Yorumları için Duygu Analizi (Turkish NLP Emotion Analysis)
Bu proje, e-ticaret sitelerinden gelen müşteri yorumlarını analiz etmek, duygu durumlarını (Olumlu/Olumsuz) belirlemek ve konularına göre (Kargo, Ürün, Hizmet vb.) sınıflandırmak amacıyla geliştirilmiş bir NLP (Doğal Dil İşleme) uygulamasıdır.

## Projenin Amacı
Müşteri geri bildirimlerini manuel olarak okumak yerine; **Yapay Zeka** ve **Kural Tabanlı** algoritmaları birleştirerek otomatik, hızlı ve ölçülebilir bir raporlama sistemi oluşturmaktır.

## Kullanılan Teknolojiler ve Yöntemler
* **Python**
* **Hugging Face Transformers:** Türkçe için eğitilmiş `savasy/bert-base-turkish-sentiment-cased` modeli kullanıldı.
* **Pandas:** Veri manipülasyonu ve raporlama için kullanıldı.
* **Hibrit Mimari:**
    * *Sentiment:* Derin öğrenme (Deep Learning) modeli ile duygu tespiti.
    * *Category:* Anahtar kelime tabanlı (Rule-Based) konu tespiti.

## Nasıl Çalışır?
1. **Veri Simülasyonu:** API'den JSON formatında gelen ham veri simüle edilir.
2. **Preprocessing (Ön İşleme):** Metinler temizlenir (Küçük harf dönüşümü, noktalama temizliği, trimming).
3. **AI Analizi:** Temizlenen metin BERT modeline verilir ve duygu skoru hesaplanır.
4. **Kategorizasyon:** Yorumun içeriğine göre (Örn: "kargo", "iade") otomatik etiket atanır.
5. **Raporlama:** Sonuçlar birleştirilerek analiz tablosu oluşturulur.

## Örnek Çıktı

| Yorum Özeti                 | Yapay Zeka Kararı | Güven Skoru | Kategori            |
|-----------------------------|-------------------|-------------|---------------------|
| "Kargo çok geç geldi!!"     | OLUMSUZ           | 0.98        | Lojistik / Kargo    |
| "Fiyat performans ürünü..." | OLUMLU            | 0.99        | Fiyat/Performans    |
| "İade etmek zorunda kaldım" | OLUMSUZ           | 0.95        | Müşteri Hizmetleri  |

