Bu proje, "Makine_verileri.xlsx" dosyasını kullanarak çeşitli veri işleme ve model değerlendirme adımlarını gerçekleştiren bir makine öğrenmesi projesidir.

## Başlarken

Bu bölüm, projenin nasıl çalıştırılacağı ve gereken kurulum adımları hakkında bilgi verir.

### Gereksinimler

Projenin çalışması için aşağıdaki Python kütüphanelerinin yüklü olması gerekmektedir:

- pandas
- ydata_profiling
- sklearn (scikit-learn)
- imbalanced-learn (imblearn)
- numpy
- xgboost
- matplotlib
- seaborn
- joblib

### Kurulum

Proje dizininde `app.py` dosyasını çalıştırmadan önce aşağıdaki adımları izleyin:

1. Bu projeyi yerel makinenize klonlayın:

   ```bash
   git clone <repository-url>
   ```

2. Gerekli Python kütüphanelerini yükleyin:

   ```bash
   pip install pandas ydata_profiling scikit-learn imbalanced-learn numpy xgboost matplotlib seaborn joblib
   ```

3. Model dosyasını indirerek projenin ana dizinine kaydedin.

### Çalıştırma

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. Ana dizinde `app.py` dosyasını çalıştırın:

   ```bash
   python app.py
   ```

Bu komut, "Makine_verileri.xlsx" dosyasını kullanarak veri işleme ve model değerlendirme adımlarını gerçekleştirir.

## Uygulama Akışı

`app.py` dosyası, `Main` sınıfını kullanarak aşağıdaki adımları gerçekleştirir:

1. Kolon tiplerinin değiştirilmesi (`change_column_type`)
2. Kodlama işlemi (`encoding`)
3. Tarih sütununun bölünmesi (`split_date`)
4. İstatistiksel anormalliklerin kapatılması (`cap_numerical_outliers_IQR`)
5. Verilerin ölçeklendirilmesi (`scaled_data`)
6. Zaman penceresi özelliklerinin oluşturulması (`create_time_window_features`)
7. Verilerin bölünmesi (`split_data`)
8. PCA uygulanması (`apply_pca`)
9. Modelin yüklenmesi (`load_model`)
10. Modelin değerlendirilmesi (`evaluate_model`)

## Model Çıktıları

Bu projede, tahmin edilen sınıflar aşağıdaki gibi yorumlanır:

- **0**: Arıza durumu
- **1**: Arıza olmama durumu

## Sonuçlar

Modelin performans sonuçları aşağıdaki gibidir:

- **F1 Score**: 0.9347
- **Recall Score**: 0.8783
- **ROC AUC Score**: 0.8281
- **Confusion Matrix**:
      |  | 0 | 1 |
      | --- | --- | --- |
      | 0 | 7 | 2 |
      | 1 | 219 | 1581 |
- **Balanced Accuracy Score**: 0.8281

## Yazar

- **Eren Coşkun**
