# src/mlflow_ui.py
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import urllib.request

from .config import ROOT, MLFLOW_DB_PATH, MLFLOW_ARTIFACTS_DIR


def _as_sqlite_uri(db_path: Path) -> str:
    # ЗАЛИШАЄМО як у тебе працювало вручну, але абсолютний шлях
    return f"sqlite:///{db_path.resolve().as_posix()}"


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _is_http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.7) as r:
            return 200 <= r.status < 500
    except Exception:
        return False


def _tail_text(path: Path, max_chars: int = 6000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return txt[-max_chars:] if len(txt) > max_chars else txt
    except Exception:
        return ""


def start_mlflow_ui(
    host: str = "127.0.0.1",
    port: int = 5000,
    open_browser: bool = True,
    quiet: bool = True,
    wait: bool = False,
    startup_timeout_sec: float = 120.0,  # <- на Windows реально треба більше
):
    url = f"http://{host}:{port}"

    # Якщо вже хтось слухає порт — просто відкриємо
    if _is_port_open(host, port):
        print(f"MLflow UI вже запущений: {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return None

    backend_uri = _as_sqlite_uri(MLFLOW_DB_PATH)

    # Важливо: так само як у твоїй ручній команді (PATH, не URI)
    artifact_root = str(MLFLOW_ARTIFACTS_DIR.resolve())

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_uri,
        "--default-artifact-root",
        artifact_root,
        "--host",
        host,
        "-p",
        str(port),
    ]

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mlflow_ui.log"

    # КЛЮЧОВЕ: перезаписуємо лог щоразу (щоб не підхоплювати старі хвости)
    stdout = None
    stderr = None
    log_file = None
    if quiet:
        log_file = open(log_path, "w", encoding="utf-8", errors="ignore")
        stdout = log_file
        stderr = log_file

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )

    deadline = time.time() + startup_timeout_sec

    # Чекаємо поки UI реально почне відповідати по HTTP
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            break

        # 1) спочатку порт
        if _is_port_open(host, port):
            # 2) потім реальна відповідь по HTTP
            if _is_http_ready(url):
                print(f"MLflow UI: {url}")
                if open_browser:
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass

                if wait:
                    try:
                        proc.wait()
                    except KeyboardInterrupt:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                return proc

        time.sleep(0.3)

    # Якщо ми тут — або процес помер, або так і не піднявся
    if log_file:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    rc = proc.poll()
    print("MLflow UI НЕ запустився.")
    if rc is not None:
        print(f"Процес завершився одразу. Return code: {rc}")
    else:
        print("Процес не дав HTTP-відповідь у заданий час (може завис/заблокований).")

    print(f"Лог: {log_path}")
    tail = _tail_text(log_path)
    if tail:
        print("\nОстанні рядки логу:\n" + tail)

    # На випадок зависання — прибиваємо, щоб не висів у фоні
    try:
        proc.terminate()
    except Exception:
        pass

    return None