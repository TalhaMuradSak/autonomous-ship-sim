"""
egitim_scripti.py

gemi_rl_ortami.py'deki GemiRLOrtami uzerinde stable-baselines3 SAC ile
egitim. Once gz-sim'in ACIK ve PLAY durumunda oldugundan emin ol.

ONEMLI: gz-sim gercek zamanli calisiyor (RTF ~%80-100). Her episode MAKS_ADIM
(1000) * ADIM_SURESI (0.5s) = ~500 saniye surebilir (episode erken biterse
daha kisa). Bu yuzden RL egitimi COK yavas olacak - birkac bin timestep bile
saatler surebilir. Once kucuk bir TOPLAM_TIMESTEP (ör. 5000-10000) ile
egitimin gercekten calistigini, odul egrisinin anlamli bir sey yaptigini
gor, sonra suresini kademeli artir.

Egitimi gz-sim penceresini gormeden, arka planda daha hizli calistirmak
istersen `gz sim -s ...world.sdf` (server-only, GUI'siz) modunu
arastirmani oneririm - bu GUI render yukunu kaldirip RTF'i artirabilir.
"""

import os
import argparse
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gemi_rl_ortami import GemiRLOrtami


# ── Hedef noktalari ────────────────────────────────────────────────────────
# otonom_gemi.py'deki WAYPOINTS ile ayni dortgen rota - karsilastirma icin
# tutarli olsun diye
HEDEF_NOKTALARI = [
    (50.0, 0.0),
    (50.0, 30.0),
    (0.0, 30.0),
    (0.0, 0.0),
]

CIKTI_KOKU = "rl_egitim_ciktilari"


def ortam_olustur():
    env = GemiRLOrtami(hedef_noktalari=HEDEF_NOKTALARI)
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000,
                         help="Toplam egitim timestep sayisi (varsayilan: 10000, kucuk bir ilk test icin)")
    parser.add_argument("--checkpoint-freq", type=int, default=2_000,
                         help="Kac timestep'te bir checkpoint kaydedilsin")
    parser.add_argument("--devam-et", type=str, default=None,
                         help="Var olan bir .zip checkpoint dosyasindan egitime devam et")
    args = parser.parse_args()

    zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
    calisma_dizini = os.path.join(CIKTI_KOKU, zaman_damgasi)
    checkpoint_dizini = os.path.join(calisma_dizini, "checkpoints")
    tensorboard_dizini = os.path.join(calisma_dizini, "tensorboard")
    os.makedirs(checkpoint_dizini, exist_ok=True)
    os.makedirs(tensorboard_dizini, exist_ok=True)

    print(f"Cikti dizini: {calisma_dizini}")
    print("Ortam olusturuluyor (gz-sim'in ACIK ve PLAY durumunda oldugundan emin ol)...")

    env = DummyVecEnv([ortam_olustur])

    if args.devam_et:
        print(f"Checkpoint'ten devam ediliyor: {args.devam_et}")
        model = SAC.load(args.devam_et, env=env, tensorboard_log=tensorboard_dizini)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_dizini,
            learning_rate=3e-4,
            buffer_size=50_000,   # gz-sim yavas oldugu icin cok buyuk tutmaya gerek yok
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            learning_starts=500,  # ilk 500 step'te rastgele aksiyon, sonra ogrenmeye basla
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dizini,
        name_prefix="gemi_sac",
    )

    print(f"Egitim basliyor: {args.timesteps} timestep")
    print(f"TensorBoard icin: tensorboard --logdir {tensorboard_dizini}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nEgitim kullanici tarafindan durduruldu, mevcut model kaydediliyor...")

    son_model_yolu = os.path.join(calisma_dizini, "son_model.zip")
    model.save(son_model_yolu)
    print(f"Model kaydedildi: {son_model_yolu}")

    env.close()


if __name__ == "__main__":
    main()
