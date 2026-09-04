# Gemi Simulasyonu (Otonom Deniz Araci)

Gazebo Sim (gz-sim) 10.4.0 (Harmonic/Jetty) uzerinde gelistirilen, 20 metrelik gercekci bir deniz aracinin otonom navigasyonu icin yapilan simulasyon projesi. Amac, engel kacinma ve waypoint takibini birlestiren bir otonom navigasyon ajani egitmek.

## Proje Yapisi

gemi_sim_projesi/
  gazebo/models/gemi_modeli/   - SDF model, world ve mesh dosyalari
  scripts/dalga_efekti.py      - Dalga kuvveti enjeksiyonu
  scripts/egitim_scripti.py    - SAC (Stable-Baselines3) egitim scripti
  scripts/gemi_rl_ortami.py    - Gymnasium RL ortami (gz-sim wrapper)
  scripts/otonom_gemi.py       - Pure Pursuit tabanli waypoint navigasyon kontrolcusu
  README.md

## Gereksinimler

- Gazebo Sim (gz-sim) 10.4.0 (Harmonic/Jetty)
- Python 3 + Gymnasium, Stable-Baselines3, gz-transport / gz-msgs Python binding'leri
- CUDA destekli GPU (RL egitimi icin onerilir)
- WSL (Ubuntu) + Mesa GPU render ortami

## Mevcut Durum

- Gemi modeli, mesh tabanli 13 katmanli collision geometrisiyle buoyancy acisindan stabil.
- Pervane (thruster) ve dumen (+-35 derece, Z ekseni revolute joint) kontrolu calisir durumda.
- GPU lidar (180 derece ileri, 100m menzil) ve GPS/navsat sensorleri aktif.
- Dalga, ruzgar ve akinti gibi cevresel etkiler simule ediliyor.
- Pure Pursuit tabanli waypoint takip kontrolcusu (otonom_gemi.py) dogrulandi.
- SAC tabanli RL egitim altyapisi kuruldu, ilk egitim kosulari tamamlandi.

## Kurulum ve Calistirma

(Buraya kendi kurulum adimlarini ekleyebilirsin - ornegin model dosyalarinin GZ_SIM_RESOURCE_PATH'e nasil eklenecegi, world'un nasil baslatilacagi vb.)

## Notlar

- Turkce isimlendirme kullanilmistir (govde = hull, pervane = propeller, dumen = rudder).
- Dumen komutlarinda pozitif aci negatif/CW yaw urettigi icin send_rudder() icinde aci negatif olarak gonderilir.
