"""
gemi_rl_ortami.py

gz-sim 10.4.0 (Jetty) uzerinde calisan gemi_modeli icin gymnasium.Env sarmalayicisi.
Engelden kacinma + waypoint takibi RL egitimi icin temel iskelet.

ONEMLI: Bu bir ISKELET dosyadir. Asagidaki TODO isaretli yerler senin projenin
gercek topic/servis isimlerine ve mevcut navigasyon kontrolcundeki pattern'lere
(navsat/pose icin ayri thread, subprocess ile "gz model -m ... -p") gore
tamamlanmali. Once reset() mekanizmasini gz-sim'de manuel test etmeni oneririm,
cunku RL egitiminde en cok sorun cikan kisim burasi.
"""

import subprocess
import threading
import time
import re
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from gz.transport import Node
    from gz.msgs.laserscan_pb2 import LaserScan
    from gz.msgs.navsat_pb2 import NavSat
    from gz.msgs.double_pb2 import Double
except ImportError as e:
    raise ImportError(
        f"gz.transport/gz.msgs import edilemedi ({e}) - "
        "'python3 -c \"import gz; print(gz.__path__)\"' ile kurulu paketleri kontrol et"
    )


REFERANS_ENLEM_DEG = 41.015137
REFERANS_BOYLAM_DEG = 28.979530
DUNYA_YARICAPI_M = 6378137.0  # WGS84 - otonom_gemi.py'deki ile ayni


def enlem_boylam_to_xy(enlem_deg, boylam_deg):
    """otonom_gemi.py / pose_listener() ile birebir ayni formul."""
    x = (boylam_deg - REFERANS_BOYLAM_DEG) * math.pi / 180.0 * DUNYA_YARICAPI_M * math.cos(math.radians(REFERANS_ENLEM_DEG))
    y = (enlem_deg - REFERANS_ENLEM_DEG) * math.pi / 180.0 * DUNYA_YARICAPI_M
    return x, y


DUNYA_ADI = "gemi_dunyasi"
MODEL_ADI = "gemi_modeli"

LIDAR_TOPIC = f"/model/{MODEL_ADI}/lidar"
NAVSAT_TOPIC = f"/model/{MODEL_ADI}/navsat"
THRUST_TOPIC = f"/model/{MODEL_ADI}/joint/pervane_mili/cmd_thrust"
RUDDER_TOPIC = f"/model/{MODEL_ADI}/joint/dumen_mili/0/cmd_pos"

MAX_THRUST = 40000.0       # N - mevcut navigasyon kontrolcunle ayni
MAX_RUDDER = 0.6109         # rad, otonom_gemi.py ile birebir ayni deger
LIDAR_MAX_RANGE = 100.0    # sensor tanimindaki <max> ile ayni
LIDAR_SAMPLES = 180        # sensor tanimindaki <samples> ile ayni

ADIM_SURESI = 0.5          # saniye - her RL step'inde ne kadar simulasyon zamani gececek
MAKS_ADIM = 1000           # episode basina maksimum step sayisi
CARPISMA_ESIGI = 3.0       # metre - bu mesafenin altinda carpisma sayilir
HEDEFE_ULASMA_ESIGI = 5.0  # metre


class GemiRLOrtami(gym.Env):
    """
    Observation (toplam LIDAR_SAMPLES + 5 boyut):
        [0 : LIDAR_SAMPLES]      -> normalize edilmis lidar mesafeleri (0-1)
        [LIDAR_SAMPLES + 0]      -> hedefe olan mesafe (normalize)
        [LIDAR_SAMPLES + 1]      -> hedefe gore bearing (sin)
        [LIDAR_SAMPLES + 2]      -> hedefe gore bearing (cos)
        [LIDAR_SAMPLES + 3]      -> mevcut hiz (normalize)
        [LIDAR_SAMPLES + 4]      -> mevcut yaw rate (normalize)

    Action (2 boyut, [-1, 1] araliginda, step() icinde olcekleniyor):
        [0] -> pervane thrust komutu
        [1] -> dumen aci komutu
    """

    metadata = {"render_modes": []}

    def __init__(self, hedef_noktalari):
        super().__init__()

        self.hedef_noktalari = hedef_noktalari  # [(x1,y1), (x2,y2), ...]
        self.mevcut_hedef_index = 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(LIDAR_SAMPLES + 5,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self._node = Node()
        self._lock = threading.Lock()

        self._son_lidar = np.full(LIDAR_SAMPLES, LIDAR_MAX_RANGE, dtype=np.float32)
        self._son_konum = np.array([0.0, 0.0])   # x, y (navsat'tan)
        self._son_yaw = 0.0                       # gz model -p'den
        self._son_hiz = 0.0
        self._son_yaw_rate = 0.0
        self._onceki_yaw = 0.0
        self._onceki_zaman = time.time()

        self._thrust_pub = self._node.advertise(THRUST_TOPIC, Double)
        self._rudder_pub = self._node.advertise(RUDDER_TOPIC, Double)

        self._node.subscribe(LaserScan, LIDAR_TOPIC, self._lidar_callback)
        self._node.subscribe(NavSat, NAVSAT_TOPIC, self._navsat_callback)

        # TODO: mevcut navigasyon kontrolcundeki gibi ayri bir pose-okuma thread'i baslat
        # (gz model -m gemi_modeli -p subprocess cagrisi ile yaw okunuyor)
        self._pose_thread_calisiyor = True
        self._pose_proc = None
        self._pose_thread = threading.Thread(target=self._pose_okuma_dongusu, daemon=True)
        self._pose_thread.start()

        self._adim_sayaci = 0

    # ------------------------------------------------------------------
    # Callback'ler
    # ------------------------------------------------------------------

    def _lidar_callback(self, msg):
        with self._lock:
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, posinf=LIDAR_MAX_RANGE, neginf=0.0)
            self._son_lidar = ranges

    def _navsat_callback(self, msg):
        with self._lock:
            x, y = enlem_boylam_to_xy(msg.latitude_deg, msg.longitude_deg)
            self._son_konum = np.array([x, y])
            self._son_hiz = float(np.hypot(msg.velocity_east, msg.velocity_north))

    def _pose_okuma_dongusu(self):
        """
        otonom_gemi.py / yaw_listener() ile birebir ayni yaklasim: gz model -p
        yerine /world/{world}/pose/info topic'i surekli dinlenir (Popen ile
        stream edilir), gemi_modeli bloguna girilince orientation quaternion'dan
        (z, w) regex ile cikarilip yaw = 2*atan2(z, w) hesaplanir.
        """
        proc = subprocess.Popen(
            ["gz", "topic", "-e", "-t", f"/world/{DUNYA_ADI}/pose/info"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._pose_proc = proc
        buf = []
        in_model = False
        try:
            for line in proc.stdout:
                if not self._pose_thread_calisiyor:
                    break
                s = line.strip()
                if f'name: "{MODEL_ADI}"' in s:
                    in_model = True
                    buf = [s]
                    continue
                if in_model:
                    buf.append(s)
                    if len(buf) > 15:
                        block = " ".join(buf)
                        ori = re.search(r'orientation\s*\{([^}]*)\}', block)
                        if ori:
                            ob = ori.group(1)
                            mz = re.search(r'z:\s*([-\d.e+]+)', ob)
                            mw = re.search(r'w:\s*([-\d.e+]+)', ob)
                            z_q = float(mz.group(1)) if mz else 0.0
                            w = float(mw.group(1)) if mw else 1.0
                            yeni_yaw = 2.0 * math.atan2(z_q, w)

                            simdi = time.time()
                            dt = max(simdi - self._onceki_zaman, 1e-3)
                            with self._lock:
                                self._son_yaw_rate = (yeni_yaw - self._onceki_yaw) / dt
                                self._son_yaw = yeni_yaw
                            self._onceki_yaw = yeni_yaw
                            self._onceki_zaman = simdi
                        in_model = False
                        buf = []
        finally:
            proc.terminate()

    # ------------------------------------------------------------------
    # Gymnasium arayuzu
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._adim_sayaci = 0
        self.mevcut_hedef_index = 0

        self._sifir_hiza_getir()
        self._model_pozisyonunu_ayarla(MODEL_ADI, 0.0, 0.0, -0.1091)
        self._engelleri_rastgele_yerlestir()

        time.sleep(0.5)  # simulasyonun oturmasini, lidar/navsat'in guncellenmesini bekle
        gozlem = self._gozlem_olustur()
        return gozlem, {}

    def _model_pozisyonunu_ayarla(self, model_adi, x, y, z, yaw=0.0):
        req = (
            f'name: "{model_adi}", '
            f'position: {{x: {x} y: {y} z: {z}}}, '
            f'orientation: {{w: {np.cos(yaw / 2)} z: {np.sin(yaw / 2)}}}'
        )
        subprocess.run(
            ["gz", "service", "-s", f"/world/{DUNYA_ADI}/set_pose",
             "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
             "--timeout", "2000", "--req", req],
            capture_output=True, timeout=3.0
        )

    def _sifir_hiza_getir(self, maks_bekleme=6.0, hiz_esigi=0.05):
        """
        Onemli bulgu: cmd_thrust=0 hizi aninda sifirlamiyor, hidrodinamik damping
        ile ustel olarak sonduruyor (tau ~2.6s). RL episode'lari arasinda 13s
        beklemek pratik degil, o yuzden once kisa bir ters-itki (fren) uyguluyoruz,
        sonra thrust=0'a geciyoruz ve navsat'tan hizin gercekten dustugunu polling
        ile dogruluyoruz.
        """
        self._rudder_pub.publish(Double(data=0.0))

        with self._lock:
            mevcut_hiz = self._son_hiz

        # kisa firenleme: mevcut hizla ters yonde kucuk bir itki (asiri
        # olmasin diye MAX_THRUST'in kucuk bir orani)
        if mevcut_hiz > hiz_esigi:
            self._thrust_pub.publish(Double(data=-0.15 * MAX_THRUST))
            time.sleep(0.8)

        self._thrust_pub.publish(Double(data=0.0))

        baslangic = time.time()
        while time.time() - baslangic < maks_bekleme:
            with self._lock:
                hiz = self._son_hiz
            if hiz < hiz_esigi:
                break
            time.sleep(0.1)
        # TODO: maks_bekleme asilirsa (hiz hala esigin ustundeyse) burada bir
        # uyari loglamayi dusun - RL egitiminde sessizce gecmesi episode'lari
        # sessizce kirletebilir

    def _engelleri_rastgele_yerlestir(self):
        engel_isimleri = ["samandira_1", "engel_tekne_1", "mendirek_1"]
        for isim in engel_isimleri:
            x = self.np_random.uniform(20, 150)
            y = self.np_random.uniform(-25, 25)
            self._model_pozisyonunu_ayarla(isim, x, y, 0.0)

    def step(self, action):
        thrust_komutu = float(np.clip(action[0], -1, 1)) * MAX_THRUST
        # otonom_gemi.py / send_rudder(): pozitif komut -> negatif (CW) yaw
        # dogrulanmis bulgu, isaret burada da ters cevriliyor
        rudder_komutu = -float(np.clip(action[1], -1, 1)) * MAX_RUDDER

        self._thrust_pub.publish(Double(data=thrust_komutu))
        self._rudder_pub.publish(Double(data=rudder_komutu))

        time.sleep(ADIM_SURESI)
        self._adim_sayaci += 1

        gozlem = self._gozlem_olustur()
        odul, carpisti, hedefe_ulasti = self._odul_hesapla()

        sonlandi = carpisti or hedefe_ulasti
        kesildi = self._adim_sayaci >= MAKS_ADIM

        return gozlem, odul, sonlandi, kesildi, {"carpisti": carpisti, "hedefe_ulasti": hedefe_ulasti}

    # ------------------------------------------------------------------
    # Yardimci fonksiyonlar
    # ------------------------------------------------------------------

    def _gozlem_olustur(self):
        with self._lock:
            lidar_norm = np.clip(self._son_lidar / LIDAR_MAX_RANGE, 0, 1)
            konum = self._son_konum.copy()
            yaw = self._son_yaw
            yaw_rate = self._son_yaw_rate

        hx, hy = self.hedef_noktalari[self.mevcut_hedef_index]
        dx, dy = hx - konum[0], hy - konum[1]
        mesafe = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - yaw

        gozlem = np.concatenate([
            lidar_norm,
            [
                np.clip(mesafe / 200.0, 0, 1),
                np.sin(bearing),
                np.cos(bearing),
                np.clip(self._son_hiz / 10.0, -1, 1),
                np.clip(yaw_rate / 2.0, -1, 1),
            ]
        ]).astype(np.float32)

        return gozlem

    def _odul_hesapla(self):
        with self._lock:
            min_lidar = float(np.min(self._son_lidar))
            konum = self._son_konum.copy()

        hx, hy = self.hedef_noktalari[self.mevcut_hedef_index]
        mesafe = np.hypot(hx - konum[0], hy - konum[1])

        carpisti = min_lidar < CARPISMA_ESIGI
        hedefe_ulasti = mesafe < HEDEFE_ULASMA_ESIGI

        odul = -0.01                       # her adimda kucuk zaman cezasi
        odul += -mesafe * 0.01             # hedefe uzaklik cezasi (yakinlik tesvigi)

        if min_lidar < CARPISMA_ESIGI * 3:
            odul -= (CARPISMA_ESIGI * 3 - min_lidar) * 0.5  # engele yaklasma cezasi

        if carpisti:
            odul -= 100.0
        if hedefe_ulasti:
            odul += 100.0
            if self.mevcut_hedef_index < len(self.hedef_noktalari) - 1:
                self.mevcut_hedef_index += 1
                hedefe_ulasti = False  # son hedef degilse episode devam etsin

        return odul, carpisti, hedefe_ulasti

    def close(self):
        self._pose_thread_calisiyor = False
        if self._pose_proc is not None:
            self._pose_proc.terminate()
