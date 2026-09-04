#!/usr/bin/env python3
import subprocess, time, math, sys, threading

WORLD = "gemi_dunyasi"
MODEL = "gemi_modeli"

THRUST_TOPIC = f"/model/{MODEL}/joint/pervane_mili/cmd_thrust"
RUDDER_TOPIC = f"/model/{MODEL}/joint/dumen_mili/0/cmd_pos"

WAYPOINTS = [
    (50.0,  0.0),
    (50.0, 30.0),
    ( 0.0, 30.0),
    ( 0.0,  0.0),
]

MAX_THRUST         = 40000.0   # artırıldı
BRAKE_THRUST       = -15000.0
APPROACH_DIST      = 20.0
ARRIVAL_DIST       = 4.0
RUDDER_GAIN        = 0.6
CONTROL_RATE_HZ    = 10.0
TURN_THRESHOLD_DEG = 20.0
SPEED_ZERO_THRESH  = 0.08


# ── Paylaşılan durum ──────────────────────────────────────────────────────────
pose_state = {"x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 0.0, "ready": False}
pose_lock  = threading.Lock()


def pose_listener():
    """navsat topic'inden konum + hız oku."""
    proc = subprocess.Popen(
        ["gz", "topic", "-e", "-t", f"/model/{MODEL}/navsat"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    lat, lon, ve, vn = None, None, 0.0, 0.0
    # Referans koordinatları (world.sdf'teki spherical_coordinates ile aynı)
    LAT0 = 41.015137
    LON0 = 28.979530
    R    = 6378137.0   # WGS84 yarıçapı (m)

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("latitude_deg:"):
            try: lat = float(line.split(":")[1])
            except: pass
        elif line.startswith("longitude_deg:"):
            try: lon = float(line.split(":")[1])
            except: pass
        elif line.startswith("velocity_east:"):
            try: ve = float(line.split(":")[1])
            except: pass
        elif line.startswith("velocity_north:"):
            try: vn = float(line.split(":")[1])
            except: pass

            if lat is not None and lon is not None:
                # GPS → yerel X/Y metre dönüşümü
                x = (lon - LON0) * math.pi / 180.0 * R * math.cos(math.radians(LAT0))
                y = (lat - LAT0) * math.pi / 180.0 * R
                spd = math.hypot(ve, vn)
                with pose_lock:
                    pose_state["x"]     = x
                    pose_state["y"]     = y
                    pose_state["speed"] = spd
                    pose_state["ready"] = True


def yaw_listener():
    """pose/info topic'inden yaw oku — sadece yaw için."""
    proc = subprocess.Popen(
        ["gz", "topic", "-e", "-t", f"/world/{WORLD}/pose/info"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    buf = []
    in_model = False
    for line in proc.stdout:
        s = line.strip()
        if f'name: "{MODEL}"' in s:
            in_model = True
            buf = [s]
            continue
        if in_model:
            buf.append(s)
            if len(buf) > 15:
                block = " ".join(buf)
                # orientation bloğundan w ve z bul
                import re
                ori = re.search(r'orientation\s*\{([^}]*)\}', block)
                if ori:
                    ob = ori.group(1)
                    mz = re.search(r'z:\s*([-\d.e+]+)', ob)
                    mw = re.search(r'w:\s*([-\d.e+]+)', ob)
                    z_q = float(mz.group(1)) if mz else 0.0
                    w   = float(mw.group(1)) if mw else 1.0
                    yaw = 2.0 * math.atan2(z_q, w)
                    with pose_lock:
                        pose_state["yaw"] = yaw
                in_model = False
                buf = []


def get_state():
    with pose_lock:
        return (pose_state["x"], pose_state["y"],
                pose_state["yaw"], pose_state["speed"])


# ── Komut gönderme ────────────────────────────────────────────────────────────

def send_thrust(n):
    n = max(-MAX_THRUST, min(MAX_THRUST, n))
    subprocess.run(
        ["gz", "topic", "-t", THRUST_TOPIC, "-m", "gz.msgs.Double",
         "-p", f"data: {n:.1f}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def send_rudder(angle):
    # DÜZELTİLDİ: pozitif komut → negatif yaw (sağa dönüş) olduğu doğrulandı
    # Bu yüzden işareti ters çeviriyoruz
    angle = max(-0.6109, min(0.6109, -angle))
    subprocess.run(
        ["gz", "topic", "-t", RUDDER_TOPIC, "-m", "gz.msgs.Double",
         "-p", f"data: {angle:.4f}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop():
    send_thrust(0.0)
    subprocess.run(
        ["gz", "topic", "-t", RUDDER_TOPIC, "-m", "gz.msgs.Double",
         "-p", "data: 0.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def normalize_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ── Ana kontrolcü ─────────────────────────────────────────────────────────────

def main():
    dt = 1.0 / CONTROL_RATE_HZ
    turn_threshold = math.radians(TURN_THRESHOLD_DEG)

    # Listener thread'leri başlat
    threading.Thread(target=pose_listener, daemon=True).start()
    threading.Thread(target=yaw_listener,  daemon=True).start()

    print("Otonom gemi kontrolcüsü başlatılıyor...")
    print(f"Toplam {len(WAYPOINTS)} waypoint:")
    for i, (wx, wy) in enumerate(WAYPOINTS):
        print(f"  {i+1}. ({wx:.1f}, {wy:.1f})")
    print()

    print("Pose verisi bekleniyor...", end="", flush=True)
    while True:
        with pose_lock:
            if pose_state["ready"]: break
        time.sleep(0.1)
    x, y, yaw, spd = get_state()
    print(f" OK → x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}°\n")

    try:
        for wp_idx, (target_x, target_y) in enumerate(WAYPOINTS):
            print(f"► Waypoint {wp_idx+1}/{len(WAYPOINTS)}: ({target_x:.1f}, {target_y:.1f})")

            while True:
                x, y, yaw, spd = get_state()
                dx, dy = target_x - x, target_y - y
                dist   = math.hypot(dx, dy)

                if dist < ARRIVAL_DIST:
                    print(f"\n  ✓ Ulaşıldı (x={x:.1f} y={y:.1f} dist={dist:.1f}m)")
                    # Fren: hız sıfırlanana kadar
                    print("  [FREN]", end="", flush=True)
                    while get_state()[3] > SPEED_ZERO_THRESH:
                        send_thrust(BRAKE_THRUST)
                        time.sleep(0.05)
                    send_thrust(0.0)
                    print(" durdu.")
                    time.sleep(0.5)
                    break

                target_yaw = math.atan2(dy, dx)
                yaw_error  = normalize_angle(target_yaw - yaw)
                rudder_cmd = RUDDER_GAIN * yaw_error  # send_rudder içinde işaret çevrilecek

                if abs(yaw_error) > turn_threshold:
                    if spd > SPEED_ZERO_THRESH:
                        thrust_cmd = BRAKE_THRUST
                        mod = "FRENLEME"
                    else:
                        thrust_cmd = 5000.0
                        mod = "DÖNÜYOR "
                else:
                    thrust_factor = min(1.0, dist / APPROACH_DIST)
                    thrust_cmd    = MAX_THRUST * thrust_factor
                    mod = "GİDİYOR "

                send_rudder(rudder_cmd)
                send_thrust(thrust_cmd)

                print(
                    f"  [{mod}] x={x:7.2f} y={y:7.2f} yaw={math.degrees(yaw):6.1f}°  "
                    f"dist={dist:6.1f}m  spd={spd:.2f}m/s  "
                    f"yaw_err={math.degrees(yaw_error):+6.1f}°",
                    end="\r",
                )
                time.sleep(dt)
            print()

        print("✓ Tüm waypoint'ler tamamlandı.")
        stop()

    except KeyboardInterrupt:
        print("\nKullanıcı durdurdu.")
        stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
