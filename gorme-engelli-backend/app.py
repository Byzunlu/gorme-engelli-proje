from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
import io
import time
from PIL import Image, ImageEnhance
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")

sinif_isimleri_tr = {
    'bicycle': 'bisiklet',
    'bus': 'otobus',
    'car': 'araba',
    'dog': 'kopek',
    'electric pole': 'elektrik diregi',
    'motorcycle': 'motorsiklet',
    'person': 'kisi',
    'traffic signs': 'trafik isareti',
    'tree': 'agac',
    'uncovered manhole': 'acik rogar'
}

# --- Güvenilir eşikler ---
# Araba/motorsiklet için yüksek eşik → false positive azalır
# Ağaç için daha makul bir eşik (0.20 çok düşüktü, 0.30 dene)
SINIF_CONF = {
    'car':              0.55,   # eskisi 0.35 → çok fazla false positive
    'bus':              0.50,   # eskisi 0.35
    'motorcycle':       0.55,   # eskisi 0.30 → "yok ama var" sorunu buradan
    'bicycle':          0.45,   # eskisi 0.30
    'dog':              0.45,   # eskisi 0.30
    'person':           0.40,   # eskisi 0.30
    'uncovered manhole':0.40,   # eskisi 0.30
    'electric pole':    0.40,   # eskisi 0.30
    'traffic signs':    0.40,   # eskisi 0.30
    'tree':             0.35,   # eskisi 0.20 → çok düşüktü, gürültü yaratıyordu
}

# Minimum kutu alanı (görüntü alanının %'si) — çok küçük tespitler gürültüdür
MIN_ALAN_ORANI = 0.008   # %0.8'den küçük kutuları atla

ZEMIN_ENGEL = {'uncovered manhole'}
TUM_ENGELLER = set(sinif_isimleri_tr.keys()) - {'person'}

# Global cooldown yerine per-class cooldown → bir nesne uyarısı diğerini engellemesin
son_uyari_zamani = {}   # { sinif_adi: timestamp }
GLOBAL_COOLDOWN = 1.5   # aynı sınıf için minimum bekleme (sn)
FARKLI_SINIF_COOLDOWN = 0.8  # farklı sınıf için minimum bekleme (sn)

last_any_alert = 0  # herhangi bir uyarının son zamanı


def hafif_sharpen(image):
    """
    Motion deblur kaldırıldı — agresif sharpen + gaussian fark bazen YOLO'nun
    görmediği edge artefaktları oluşturuyor ve halüsinasyona yol açıyor.
    Sadece hafif contrast + unsharp mask yeterli.
    """
    img_np = np.array(image)
    # Unsharp mask: hafif netleştirme, gürültü artırmaz
    blur = cv2.GaussianBlur(img_np, (0, 0), 2.0)
    sharp = cv2.addWeighted(img_np, 1.4, blur, -0.4, 0)
    return Image.fromarray(sharp)


def iou(box1, box2):
    """İki kutu arasında IoU hesapla (manuel NMS için)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    alan1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    alan2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = alan1 + alan2 - inter
    return inter / union if union > 0 else 0


@app.route("/predict", methods=["POST"])
def predict():
    global last_any_alert

    simdi = time.time()

    if "image" not in request.files:
        return jsonify({"error": "Görüntü bulunamadı"}), 400

    # Herhangi bir uyarı verileli çok kısa süre geçtiyse cooldown döndür
    if simdi - last_any_alert < FARKLI_SINIF_COOLDOWN:
        return jsonify({"mesaj": None, "cooldown": True})

    file = request.files["image"]
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Kamera aynalı geliyorsa düzelt
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image = image.resize((640, 640))   # 416 → 640: YOLOv8s için optimal
    image = hafif_sharpen(image)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)      # eskisi 1.3, biraz düşürdük

    genislik = image.width
    yukseklik = image.height
    goruntu_alani = genislik * yukseklik

    # Yol sınırları (yatay)
    yol_sol  = genislik * 0.10
    yol_sag  = genislik * 0.90
    sol_sinir = genislik * 0.38
    sag_sinir = genislik * 0.62

    results = model(
        image,
        conf=0.25,       # model çıkışı için alt sınır — SINIF_CONF üstünde filtreleyeceğiz
        imgsz=640,
        iou=0.45,        # NMS IoU eşiği — örtüşen aynı nesne kutularını birleştir
        agnostic_nms=True,  # sınıftan bağımsız NMS → motorsiklet+bisiklet çakışmasını önler
        verbose=False
    )

    # Tüm geçerli kutuları topla
    adaylar = []
    for box in results[0].boxes:
        sinif = results[0].names[int(box.cls)].lower()
        if sinif not in TUM_ENGELLER:
            continue

        conf_degeri = float(box.conf)
        sinif_esigi = SINIF_CONF.get(sinif, 0.40)
        if conf_degeri < sinif_esigi:
            continue

        coords = box.xyxy[0].tolist()   # [x1, y1, x2, y2]
        e_x = (coords[0] + coords[2]) / 2
        e_y = (coords[1] + coords[3]) / 2

        # Yol dışı nesneleri at
        if not (yol_sol <= e_x <= yol_sag):
            continue

        # Çok küçük kutuları at (gürültü/uzak nesne)
        kutu_alani = (coords[2] - coords[0]) * (coords[3] - coords[1])
        alan_orani = kutu_alani / goruntu_alani
        if alan_orani < MIN_ALAN_ORANI:
            continue

        # Ekranın üst %20'sinde yüzen nesneler genellikle arka plan — kaldır
        # (manhole zeminde olduğu için bu filtreden muaf)
        if sinif not in ZEMIN_ENGEL and e_y < yukseklik * 0.20:
            continue

        adaylar.append({
            'sinif': sinif,
            'conf': conf_degeri,
            'e_x': e_x,
            'e_y': e_y,
            'alan_orani': alan_orani,
            'coords': coords,
        })

    if not adaylar:
        return jsonify({"mesaj": None, "cooldown": False})

    # --- Öncelik: yakındaki (büyük alan) + yüksek confidence ---
    # Sıralama: alan_orani büyük ve conf yüksek olsun
    adaylar.sort(key=lambda x: (x['alan_orani'] * 0.6 + x['conf'] * 0.4), reverse=True)

    # En iyi adayı seç, ama per-class cooldown'a takılıyorsa sonrakine geç
    secilen = None
    for aday in adaylar:
        sinif = aday['sinif']
        son_zaman = son_uyari_zamani.get(sinif, 0)
        if simdi - son_zaman >= GLOBAL_COOLDOWN:
            secilen = aday
            break

    if secilen is None:
        return jsonify({"mesaj": None, "cooldown": True})

    # --- Mesaj üret ---
    sinif      = secilen['sinif']
    e_x        = secilen['e_x']
    alan_orani = secilen['alan_orani']
    engel_tr   = sinif_isimleri_tr.get(sinif, sinif)

    if alan_orani > 0.15:
        mesafe_ifade = "çok yakında"
    elif alan_orani > 0.05:
        mesafe_ifade = "yakında"
    else:
        mesafe_ifade = "ileride"

    if sinif in ZEMIN_ENGEL:
        mesaj = f"dikkat {engel_tr} var dikkatli olun"
    elif e_x < sol_sinir:
        mesaj = f"dikkat {mesafe_ifade} solda {engel_tr} var sağa yönelin"
    elif e_x > sag_sinir:
        mesaj = f"dikkat {mesafe_ifade} sağda {engel_tr} var sola yönelin"
    else:
        mesaj = f"dikkat {mesafe_ifade} tam önünüzde {engel_tr} var durun"

    son_uyari_zamani[sinif] = simdi
    last_any_alert = simdi

    return jsonify({"mesaj": mesaj})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)