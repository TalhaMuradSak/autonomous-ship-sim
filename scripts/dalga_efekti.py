#!/usr/bin/env python3
"""
Hafif dalga efekti simülatörü - gz-sim ApplyLinkWrench üzerinden.
Gerçek dalga alanı (asv_wave_sim) yerine, govde linkine periyodik
heave kuvveti + roll/pitch torku enjekte ederek "dalgada sallanma"
hissi üretir. gz-sim 10.4.0 (Jetty) ile ek bağımlılık gerektirmez.

Kullanım:
    python3 dalga_efekti.py
Durdurmak için Ctrl+C.
"""

import subprocess
import time
import math
import sys

# --- Ayarlar -----------------------------------------------------------
WORLD = "gemi_dunyasi"                 # world.sdf'teki <world name="..."> ile aynı olmalı
LINK = "gemi_modeli::govde"            # kuvvetin uygulanacağı link

# ÖNEMLİ: /wrench (persistent olmayan) topic'i kuvveti sadece TEK bir fizik
# adımı (max_step_size=0.001s) için uyguluyor. 12Hz'de mesaj gönderdiğimizde
# kuvvet zamanın ~%1.2'sinde uygulanıyor, pratikte etkisiz kalıyor.
# /wrench/persistent ise değeri bir sonraki güncellemeye/clear'a kadar HER
# fizik adımında uyguluyor - gerçek dalga etkisi için doğru topic bu.
PERSIST_TOPIC = f"/world/{WORLD}/wrench/persistent"
CLEAR_TOPIC = f"/world/{WORLD}/wrench/clear"

# Kaba deniz durumu: birden fazla sinüs bileşeninin toplamı.
# (genlik_N, periyot_s, faz_rad) - gerçek bir dalga spektrumunun
# (Bretschneider/JONSWAP) çok basitleştirilmiş bir yaklaşımı.
WAVE_COMPONENTS = [
    (150000.0, 6.5, 0.0),
    (80000.0, 4.2, 1.7),
    (40000.0, 9.0, 3.1),
]

ROLL_GAIN = 0.35    # heave kuvvetinden roll torku türetme oranı
PITCH_GAIN = 0.15   # heave kuvvetinden pitch torku türetme oranı
ROLL_MOD_FREQ = 0.13
PITCH_MOD_FREQ = 0.11

UPDATE_RATE_HZ = 12.0   # gövdenin atalet/sönümleme zaman sabiti (~1.5s) göz
                        # önüne alındığında bu hız fazlasıyla yeterli


def compute_wrench(t: float):
    fz = 0.0
    for amp, period, phase in WAVE_COMPONENTS:
        omega = 2.0 * math.pi / period
        fz += amp * math.sin(omega * t + phase)

    # Roll/pitch, heave ile eşlenik ama yavaş değişen bir yönelim
    # bileşeniyle moduluyor, böylece tekdüze bir "trambolin" etkisi
    # yerine daha organik bir sallanma oluşuyor.
    tx = fz * ROLL_GAIN * math.sin(t * ROLL_MOD_FREQ)
    ty = fz * PITCH_GAIN * math.cos(t * PITCH_MOD_FREQ)
    return fz, tx, ty


def clear_wrench():
    entity_msg = f'name: "{LINK}", type: LINK'
    subprocess.run(
        ["gz", "topic", "-t", CLEAR_TOPIC, "-m", "gz.msgs.Entity", "-p", entity_msg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def publish_wrench(fz: float, tx: float, ty: float):
    # Üst üste binmeyi önlemek için önce mevcut persistent wrench'i temizle,
    # sonra güncel değeri gönder.
    clear_wrench()
    msg = (
        f'entity: {{name: "{LINK}", type: LINK}}, '
        f'wrench: {{force: {{x: 0, y: 0, z: {fz:.1f}}}, '
        f'torque: {{x: {tx:.1f}, y: {ty:.1f}, z: 0}}}}'
    )
    result = subprocess.run(
        ["gz", "topic", "-t", PERSIST_TOPIC, "-m", "gz.msgs.EntityWrench", "-p", msg],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[HATA] gz topic başarısız: {result.stderr.strip()}")


def main():
    dt = 1.0 / UPDATE_RATE_HZ
    t0 = time.time()
    print(f"Dalga efekti başladı -> topic: {PERSIST_TOPIC}, link: {LINK}")
    print("Durdurmak için Ctrl+C")
    try:
        while True:
            t = time.time() - t0
            fz, tx, ty = compute_wrench(t)
            publish_wrench(fz, tx, ty)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nDurduruluyor, kuvvet temizleniyor...")
        clear_wrench()
        sys.exit(0)


if __name__ == "__main__":
    main()
