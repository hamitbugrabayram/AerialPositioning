<p align="center">
  <img src="thumbnail.png" alt="Görsel Konumlama Küçük Resmi" width="600"/>
</p>

# Uydu Görüntüleri Kullanarak Görsel Konumlama

Bu depo, görsel konumlama görevi için farklı özellik eşleştirme algoritmalarını (LightGlue, SuperGlue, GIM varyantları) karşılaştırmak amacıyla tasarlanmış bir kıyaslama çerçevesi sunar. Özellikle drone görüntülerini uydu haritalarıyla karşılaştırır. Her eşleştirici için pipeline'lar, ön işleme özellikleri (yeniden boyutlandırma, perspektif düzeltme) ve metre cinsinden konumlama hatası hesaplama içerir.


## Özellikler

*   **Çoklu Eşleştiriciler:** LightGlue, SuperGlue ve GIM (DKM, RoMa, LoFTR, LightGlue varyantı) karşılaştırması.
*   **Standartlaştırılmış Pipeline'lar:** Kolay yürütme için kapsüllenmiş pipeline'lar (`src/`).
*   **Ön İşleme:** İsteğe bağlı görüntü yeniden boyutlandırma ve perspektif düzeltme (kuşbakışı görünüm simülasyonu).
*   **Metre Düzeyinde Hata:** Ground Truth ve Tahmin arasındaki Haversine mesafesi kullanılarak konumlama hatası hesaplanır.
*   **Detaylı Çıktı:** Her çift için sonuçlar (`.txt`), görselleştirmeler (`.png`) ve genel özetler (`.csv`, `.txt`) oluşturur.

## Başlarken

### 1. Gereksinimler

*   **Conda:** Ortam yönetimi için Anaconda veya Miniconda gereklidir.

### 2. Depoyu Klonlama

Gerekli eşleştirici alt modüllerini dahil etmek için bu depoyu *recursive* olarak klonlayın:

```bash
git clone --recursive https://github.com/ALFONSOBUGRA/SatelliteLocalization.git
cd SatelliteLocalization
```

Eğer --recursive olmadan klonladıysanız, depo dizininde şunu çalıştırın:
```bash
git submodule update --init --recursive
```
### 3. Ortam Kurulumu

Conda ortamını oluşturmak için INSTALL.md dosyasındaki detaylı adımları takip edin.

### 4. Veri Hazırlığı

- Drone sorgu görüntülerinizi data/query/ dizinine yerleştirin.
- photo_metadata.csv dosyasını (Filename, Latitude, Longitude ve perspektif düzeltme kullanılıyorsa oryantasyon açıları gibi sütunlar içeren) data/query/ dizinine yerleştirin.
- Uydu harita tile görüntülerinizi data/map/ dizinine yerleştirin.
- map.csv dosyasını (Filename, Top_left_lat vb. sütunlar içeren) data/map/ dizinine yerleştirin.
(Gerekli format için sağlanan örnek CSV dosyalarına bakın.)

### 5. Model Ağırlıkları

Bu depo önceden eğitilmiş model ağırlıklarını (.ckpt, .pth) içermez.

Kullanmayı planladığınız eşleştirici(ler) için gerekli ağırlıkları indirin (özellikle GIM varyantları ve LoFTR için).

Ağırlıkları erişilebilir bir konuma yerleştirin (örn. ilgili matchers/<eşleştirici_adı>/weights dizininde. Ancak bu, alt modülün gitignore'u tarafından yok sayılabilir - matchers dışında merkezi bir weights/ klasörü daha iyi olabilir).

config.yaml dosyasındaki matcher_weights bölümündeki yolları güncelleyin (örn. gim_weights_path, loftr_weights_path).

### 6. Kıyaslamayı Çalıştırma

Conda ortamınızı etkinleştirin (örn. conda activate visloc) ve çalıştırın:
```bash
python benchmark.py --config config.yaml
```
### 7. Çıktılar
Sonuçlar data/output/ içinde zaman damgalı bir alt dizine kaydedilir. Bu içerikler şunlardır:
Her sorgu görüntüsü için klasörler:
- Detaylı metriklere sahip harita karşılaştırma .txt dosyaları.
- Eşleşme görselleştirme .png dosyaları (etkinleştirilmişse ve başarılıysa).
- benchmark_summary.csv: Her sorgu için en iyi eşleşme sonuçlarının özeti.
- benchmark_stats.txt: Kıyaslama çalıştırması için genel istatistikler.
- processed_queries/ (ön işleme kullanılıyorsa): Eşleştirme için kullanılan değiştirilmiş sorgu görüntülerini içerir.


### Teşekkürler
Bu çerçeve, WildNav'da gösterilen kavramlar ve uygulamasına dayanmaktadır.
LightGlue, SuperGlue ve GIM yazarlarının mükemmel açık kaynak çalışmalarını kullanır.
