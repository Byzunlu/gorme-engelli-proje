import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

const SERVER_URL = 'http://10.70.228.79:5000/predict';
const SCAN_INTERVAL_MS = 2000;   // 1500 → 2000: sunucuya daha az yük, cooldown ile uyumlu
const REQUEST_TIMEOUT_MS = 6000;

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [resultText, setResultText] = useState('Uygulamayı başlatmak için butona basın.');
  const [isAutomatic, setIsAutomatic] = useState(false);
  const isAutomaticRef = useRef(false);
  const scanTimeoutRef = useRef(null);   // interval yerine timeout zinciri — örtüşen istek olmaz

  const speakMessage = (text) => {
    Speech.stop();
    Speech.speak(text, { language: 'tr', rate: 1.1 });
  };

  const takePictureAndPredict = async () => {
    if (!cameraRef.current || !isAutomaticRef.current) return;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      setLoading(true);

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.65,
        skipProcessing: false,
        exif: false,
        base64: false,
        mute: true,
        shutterSound: false,
      });

      // Fotoğraf çekildikten sonra hâlâ aktif mi kontrol et
      if (!isAutomaticRef.current) {
        clearTimeout(timeoutId);
        return;
      }

      const formData = new FormData();
      formData.append('image', {
        uri: photo.uri,
        name: 'photo.jpg',
        type: 'image/jpeg',
      });

      const response = await fetch(SERVER_URL, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!isAutomaticRef.current) return;

      const data = await response.json();

      if (data.mesaj) {
        setResultText(data.mesaj);
        speakMessage(data.mesaj);
      } else if (data.cooldown) {
        // Cooldown'da ekranı temizleme, son mesajı koru
        console.log('Sunucu cooldown modunda.');
      } else {
        setResultText('Yol temiz.');
        // "Yol temiz" için her seferinde konuşma — sessiz geçiş tercih edilebilir
        // speakMessage('Yol temiz.');  ← istenirse açılabilir
      }

    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        console.log('İstek zaman aşımına uğradı.');
      } else {
        console.log('İstek Hatası:', error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Timeout zinciri: bir istek bitmeden yeni istek başlamaz
  const scheduleNextScan = () => {
    if (!isAutomaticRef.current) return;
    scanTimeoutRef.current = setTimeout(async () => {
      await takePictureAndPredict();
      scheduleNextScan();
    }, SCAN_INTERVAL_MS);
  };

  const startScanning = () => {
    isAutomaticRef.current = true;
    speakMessage('Sistem başlatıldı, etraf taranıyor.');
    setResultText('Etraf taranıyor...');
    scheduleNextScan();
  };

  const stopScanning = () => {
    isAutomaticRef.current = false;
    if (scanTimeoutRef.current) {
      clearTimeout(scanTimeoutRef.current);
      scanTimeoutRef.current = null;
    }
    Speech.stop();
    speakMessage('Sistem durduruldu.');
    setResultText('Sistem kapalı.');
    setLoading(false);
  };

  useEffect(() => {
    if (isAutomatic) {
      startScanning();
    } else {
      stopScanning();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAutomatic]);

  // Komponent unmount olursa temizle
  useEffect(() => {
    return () => {
      isAutomaticRef.current = false;
      if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current);
      Speech.stop();
    };
  }, []);

  if (!permission) return <View />;

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Kamerayı kullanmak için izninize ihtiyacımız var.</Text>
        <TouchableOpacity style={styles.startButton} onPress={requestPermission}>
          <Text style={styles.text}>İzin Ver</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.mainButton, isAutomatic ? styles.stopButton : styles.startButton]}
            onPress={() => setIsAutomatic(prev => !prev)}
          >
            <Text style={styles.text}>
              {isAutomatic ? 'TARAMAYI DURDUR' : 'SİSTEMİ BAŞLAT'}
            </Text>
          </TouchableOpacity>
        </View>
      </CameraView>

      <View style={styles.resultContainer}>
        {loading && <ActivityIndicator size="small" color="#1e90ff" style={{ marginBottom: 10 }} />}
        <Text style={styles.resultText}>{resultText}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container:       { flex: 1, backgroundColor: '#000' },
  message:         { textAlign: 'center', color: '#fff', marginBottom: 20, fontSize: 16 },
  camera:          { flex: 4 },
  buttonContainer: { flex: 1, justifyContent: 'flex-end', alignItems: 'center', marginBottom: 30 },
  mainButton:      { paddingHorizontal: 40, paddingVertical: 20, borderRadius: 30, width: '85%', alignItems: 'center', elevation: 8 },
  startButton:     { backgroundColor: '#00db6a' },
  stopButton:      { backgroundColor: '#ff4757' },
  text:            { fontSize: 20, fontWeight: 'bold', color: 'white', letterSpacing: 1 },
  resultContainer: { flex: 1, backgroundColor: '#111', justifyContent: 'center', alignItems: 'center', padding: 20 },
  resultText:      { color: '#fff', fontSize: 22, textAlign: 'center', fontWeight: 'bold' },
});