# YOLO-ModelAnalyzer

YOLO-ModelAnalyzer, YOLO modellerinin performans analizini ve görselleştirmesini sağlamak için geliştirilmiş bir Python projesidir. Bu proje, kayıp eğrileri, karmaşıklık matrisi, ROC eğrisi, AUC, mAP, F1 skoru gibi birçok metriği ölçerek model performansını değerlendirmenize olanak tanır.

## Özellikler

- **Kayıp Eğrisi Çizme**: Model eğitimindeki box, class ve DFL kayıplarını çizerek eğitimin ilerlemesini izleyin.
- **Karmaşıklık Matrisi**: Gerçek etiketler ve tahminler arasında karşılaştırma yaparak doğruluğu analiz edin.
- **ROC Eğrisi ve AUC Hesaplama**: Modelin ROC eğrisini çizerek AUC değerini hesaplayın.
- **Precision-Recall Eğrisi**: Modelin hassasiyet ve geri çağırma oranlarını analiz edin.
- **mAP Hesaplama**: Ortalama doğruluk puanını (mAP) hesaplayarak modelin doğruluğunu ölçün.
- **F1 Skoru, Doğruluk, Dengelenmiş Doğruluk**: Modelin genel doğruluğunu ve sınıflandırma yeteneklerini ölçmek için farklı metrikler kullanın.
- **Cohen's Kappa ve MCC**: Modelin sınıflandırma performansını ve anlaşma düzeyini ölçmek için Cohen's Kappa ve Matthews Correlation Coefficient kullanın.
- **Dice Katsayısı**: Modelin segmentasyon ve sınıflandırma başarısını değerlendirin.

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```bash
pip install textblob matplotlib scikit-learn pandas numpy ultralytics
