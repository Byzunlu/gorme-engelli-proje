# Görme Engelliler için Engel Tanıma Sistemi

Görme engelli bireyler için geliştirilmiş, yapay zeka destekli mobil engel tanıma uygulaması.

## Proje Hakkında

Uygulama, telefon kamerası aracılığıyla gerçek zamanlı olarak çevredeki engelleri tespit eder
ve kullanıcıyı sesli uyarı + ekran bildirimi ile bilgilendirir.
Derin öğrenme tabanlı YOLOv8 modeli kullanılarak geliştirilmiştir.

## Klasör Yapısı

- `gorme-engelli-asistan/` → Expo Go ile geliştirilmiş mobil uygulama (TypeScript)
- `gorme-engelli-backend/` → Python tabanlı yapay zeka sunucusu (Flask + YOLOv8)
- `Proje2.ipynb` → Model eğitimi ve test kodları (Google Colab)

## Kurulum

### 1. Backend (Python)

Gerekli kütüphaneleri yükleyin ve sunucuyu başlatın:

pip install flask
python app.py

Sunucu varsayılan olarak `http://localhost:5000` adresinde çalışır.

### 2. Mobil Uygulama (Expo Go)

#### Adım 1 — Expo Go Uygulamasını İndirin

Telefona **Expo Go** uygulamasını indirin:
- Android: https://play.google.com/store/apps/details?id=host.exp.exponent
- iOS: https://apps.apple.com/app/expo-go/id982107779

#### Adım 2 — Bağımlılıkları Yükleyin

Terminalde proje klasörüne girin ve şu komutu çalıştırın:

cd gorme-engelli-asistan
npm install

#### Adım 3 — Uygulamayı Başlatın

npx expo start

Terminalde bir **QR kod** görünecektir.

#### Adım 4 — QR Kodu Okutun

- **Android:** Expo Go uygulamasını açın → QR kodu tara
- **iOS:** Telefon kamerasını açın → QR kodu tara → Expo Go ile aç

Uygulama telefonunuzda otomatik olarak başlayacaktır.

### 3. Model Eğitimi

`Proje2.ipynb` dosyasını bu repoda doğrudan görüntüleyebilirsiniz.

## Teknolojiler

- React Native / Expo Go
- Python / Flask
- YOLOv8
- TypeScript

  ## ⚠️ Önemli Not

Uygulamayı çalıştırmadan önce `gorme-engelli-asistan/app/(tabs)/index.tsx` dosyasını açın.
Aşağıdaki satırı bulun ve `BILGISAYAR_IP` kısmını kendi IP adresinizle değiştirin:

const SERVER_URL = 'http://BILGISAYAR_IP:5000/predict';

IP adresinizi öğrenmek için:
- Windows: cmd açın -> `ipconfig` yazın -> IPv4 Address satırındaki adresi kullanın

### Ağ Bağlantısı Hakkında
- Telefon ve bilgisayarın **aynı Wi-Fi ağına** bağlı olması zorunludur
- Okul, yurt veya kurumsal ağlarda cihazlar arası iletişim engellenebilir
-Telefondaki mobil veriyi paylaşarak da yapabilirsiniz.
