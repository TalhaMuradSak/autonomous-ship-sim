"""
degerlendirme_scripti.py

Egitilmis SAC modelini gz-sim GUI'sinde (server-only DEGIL, GUI acik) calistirip
gemi'nin waypoint'lere dogru gercekte nasil manevra yaptigini gozle izlemek icin.

ONCE gz-sim'i GUI'li modda ac (server-only DEGIL):
    gz sim ~/.gz/sim/models/gemi_modeli/world.sdf
ve PLAY butonuna bas.

Sonra bu scripti calistir:
    python3 degerlendirme_scripti.py rl_egitim_ciktilari/20260904_231507/son_model.zip

world.sdf'teki real_time_factor'u 1.0'a geri aldigindan emin ol (egitim sirasinda
0/sinirsiz yapmis olabilirsin) - yoksa gemi gozle takip edilemeyecek kadar hizli
hareket edebilir.
"""

import sys
import time

from stable_baselines3 import SAC

from gemi_rl_ortami import GemiRLOrtami


HEDEF_NOKTALARI = [
    (50.0, 0.0),
    (50.0, 30.0),
    (0.0, 30.0),
    (0.0, 0.0),
]


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python3 degerlendirme_scripti.py <model_yolu.zip> [episode_sayisi]")
        sys.exit(1)

    model_yolu = sys.argv[1]
    episode_sayisi = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Model yukleniyor: {model_yolu}")
    model = SAC.load(model_yolu)

    env = GemiRLOrtami(hedef_noktalari=HEDEF_NOKTALARI)

    for ep in range(episode_sayisi):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{episode_sayisi} basliyor")
        print(f"{'=' * 60}")

        obs, _ = env.reset()
        toplam_odul = 0.0
        adim = 0

        while True:
            # deterministic=True: rastgele keşif yok, modelin "en iyi bildiği"
            # aksiyonu secmesini sagliyoruz - gercek performansi gormek icin dogru ayar
            aksiyon, _ = model.predict(obs, deterministic=True)
            obs, odul, sonlandi, kesildi, info = env.step(aksiyon)
            toplam_odul += odul
            adim += 1

            with env._lock:
                konum = env._son_konum.copy()
                hiz = env._son_hiz

            hedef_x, hedef_y = env.hedef_noktalari[env.mevcut_hedef_index]
            print(
                f"  adim={adim:4d}  x={konum[0]:7.2f} y={konum[1]:7.2f}  "
                f"hedef=({hedef_x:.0f},{hedef_y:.0f})  hiz={hiz:.2f}m/s  "
                f"odul={odul:+.2f}  toplam={toplam_odul:+.1f}",
                end="\r",
            )

            if sonlandi or kesildi:
                print()  # satir sonu
                if info.get("carpisti"):
                    print(f"  -> CARPTI (adim {adim})")
                elif info.get("hedefe_ulasti"):
                    print(f"  -> HEDEFE ULASTI (adim {adim})")
                else:
                    print(f"  -> zaman asimi (adim {adim})")
                print(f"  Episode toplam odul: {toplam_odul:.1f}")
                break

    env.close()
    print("\nDegerlendirme tamamlandi.")


if __name__ == "__main__":
    main()
