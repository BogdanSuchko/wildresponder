from pathlib import Path
import sys

from mangum import Mangum

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app as fastapi_app  # noqa: E402

handler = Mangum(fastapi_app)
app = fastapi_app

