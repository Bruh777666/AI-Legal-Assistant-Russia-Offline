import sys
import os
import re
from threading import Thread

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QLabel, QTextEdit, QLineEdit, QPushButton,
    QGraphicsDropShadowEffect
)
from PyQt6.QtGui import (
    QPalette, QColor, QPainter, QFont, QPixmap, QIcon,
    QPainterPath, QPen, QConicalGradient, QBrush
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QRectF, QPointF

# === БАЗОВАЯ ПАПКА ПРИЛОЖЕНИЯ (и для .py, и для .exe) ===
if getattr(sys, "frozen", False):
    # Запущено как .exe (PyInstaller)
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Запущено как обычный скрипт .py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======== Llama (локальный ИИ) =========
try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except Exception:
    HAS_LLAMA = False

MODEL_PATH = os.path.join(BASE_DIR, "models", "mistral.gguf")
llm = None


def get_llm():
    """
    Ленивая инициализация модели.
    Баланс: поумнее, но без жёстких лагов.
    """
    global llm
    if not HAS_LLAMA:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    if llm is None:
        try:
            os.environ["LLAMA_LOG_LEVEL"] = "ERROR"
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=8,   # если ПК слабый — можно уменьшить до 4
                n_batch=256,
                verbose=False
            )
        except Exception:
            return None
    return llm


# ========= ИИ в отдельном потоке =========
class AIWorker(QObject):
    finished = pyqtSignal(str)

    def __init__(self, question: str, fragment: str):
        super().__init__()
        self.question = question
        self.fragment = fragment

    def run(self):
        engine = get_llm()
        if engine is None:
            self.finished.emit(
                "Локальный ИИ не настроен.\n\n"
                "Проверь, что установлен пакет llama-cpp-python\n"
                "и существует файл models/mistral.gguf рядом с программой."
            )
            return

        # ПРОМПТ БЕЗ ТЕГОВ <ОТВЕТ>/<ВОПРОС>
        prompt = f"""
Ты юридический помощник по законодательству РФ.
Отвечай простым, понятным языком, но опирайся только на реальные законы РФ.
Если нет точной информации, честно напиши, что данных недостаточно
и порекомендуй обратиться к официальным источникам
(Консультант+, pravo.gov.ru и т.п.).

Ниже может быть фрагмент закона (если найден):

Контекст закона:
{self.fragment}

Вопрос пользователя:
{self.question}

Теперь дай развёрнутый, но по делу ответ от первого лица.
Не переписывай вопрос целиком. Не повторяй текст инструкции.
Ответ:
"""

        try:
            result = engine(
                prompt,
                max_tokens=480,        # достаточно, чтобы не обрывать мысль
                temperature=0.25,
                top_p=0.96,
                top_k=60,
                repeat_penalty=1.15,   # сильнее штраф за повтор
            )
            raw = result["choices"][0]["text"]
            text = (raw or "").strip()

            # --- Чистим явный мусор в начале ---
            for prefix in [
                "Ответ:",
                "ответ:",
                "Ответ пользователя:",
                "Вопрос пользователя:",
                "Вопрос:",
            ]:
                if text.startswith(prefix):
                    text = text[len(prefix):].lstrip()

            # защита от пустого
            if not text:
                text = "ИИ не смог сформулировать ответ. Попробуйте переформулировать вопрос."

            # Если заканчивается странно — добавим многоточие
            if text and text[-1] not in ".?!…»\"'":
                text += "…"

        except Exception as e:
            text = f"Ошибка работы ИИ: {e}"

        self.finished.emit(text)


# ========= Локальные законы =========
LAWS_FILES = [
    "constitution_rf.txt",
    "gk_rf.txt",
    "uk_rf.txt",
    "koap_rf.txt",
    "law_police.txt",
    "law_consumer.txt",
]


def load_law(filename: str) -> str:
    path = os.path.join(BASE_DIR, "laws", filename)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def detect_file(q: str) -> str | None:
    q = q.lower()
    if "уголов" in q or "ук" in q:
        return "uk_rf.txt"
    if "коап" in q or "административ" in q:
        return "koap_rf.txt"
    if "гк" in q or "граждан" in q:
        return "gk_rf.txt"
    if "конституц" in q:
        return "constitution_rf.txt"
    if "полици" in q:
        return "law_police.txt"
    if "потребител" in q:
        return "law_consumer.txt"
    return None


def find_article(query: str, laws_ok: bool):
    """Пытаемся найти статью/фрагмент в законах."""
    if not laws_ok:
        return None, None

    q = query.lower()
    nums = re.findall(r"\d+", q)
    num = nums[0] if nums else None

    codex = detect_file(q)

    if num and codex:
        full = load_law(codex)
        low = full.lower()
        pattern = f"статья {num}"
        idx = low.find(pattern)
        if idx != -1:
            return codex, full[max(0, idx - 80): idx + 900]

    for f in LAWS_FILES:
        full = load_law(f)
        if not full:
            continue
        low = full.lower()
        idx = low.find(q)
        if idx != -1:
            return f, full[max(0, idx - 80): idx + 900]

    return None, None


# ========= UI-элементы =========
class Glass(QWidget):
    """Стеклянная панель с мягкой подсветкой."""
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
        QWidget{
            background: rgba(20,20,20,0.88);
            border-radius:22px;
            border:1px solid rgba(255,255,255,0.25);
        }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(255, 255, 255, 60))
        self.setGraphicsEffect(shadow)


class Btn(QPushButton):
    def __init__(self, t: str):
        super().__init__(t)
        self.setFixedHeight(46)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.setStyleSheet("""
        QPushButton{
            background: rgba(255,255,255,0.13);
            border-radius:18px;
            color:white;
        }
        QPushButton:hover{
            background: rgba(255,255,255,0.28);
        }
        QPushButton:pressed{
            background:#1D9BF0;
        }
        """)


class RainbowAvatar(QWidget):
    """Круглая аватарка с радужным ободком."""
    def __init__(self, path: str, size: int = 80, parent=None):
        super().__init__(parent)
        self.size_val = size
        self.setFixedSize(size + 12, size + 12)
        self.phase = 0.0

        self.src = QPixmap(path)
        if self.src.isNull():
            self.src = QPixmap(self.size_val, self.size_val)
            self.src.fill(Qt.GlobalColor.darkGray)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    def _tick(self):
        self.phase = (self.phase + 3) % 360
        self.update()

    def paintEvent(self, event):
        s = self.size_val
        w = float(self.width())
        h = float(self.height())

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = QPointF(w / 2.0, h / 2.0)
        radius = s / 2.0 + 4.0

        # фон — темнее фона
        p.setBrush(QColor(0, 0, 0, 230))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(center, radius, radius)

        # радужный ободок
        grad = QConicalGradient(center, float(self.phase))
        grad.setColorAt(0.0, QColor(255, 90, 90))
        grad.setColorAt(0.16, QColor(255, 160, 90))
        grad.setColorAt(0.33, QColor(255, 255, 120))
        grad.setColorAt(0.5, QColor(120, 255, 160))
        grad.setColorAt(0.66, QColor(90, 190, 255))
        grad.setColorAt(0.83, QColor(200, 130, 255))
        grad.setColorAt(1.0, QColor(255, 90, 140))

        pen = QPen(QBrush(grad), 5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center, radius - 2, radius - 2)

        # внутренняя круглая аватарка
        inner_r = radius - 6
        circle = QPainterPath()
        circle.addEllipse(center, inner_r, inner_r)
        p.setClipPath(circle)

        img = self.src.scaled(
            int(inner_r * 2),
            int(inner_r * 2),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        p.drawPixmap(int(center.x() - inner_r),
                     int(center.y() - inner_r),
                     img)

        # лёгкое затемнение внутри круга
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))

        p.end()


class RainbowLineEdit(QLineEdit):
    """
    Поле ввода:
    - фон тёмный
    - радуга по контуру
    - текст ярко-белый
    - placeholder почти белый
    - фиолетовое выделение + белый glow по контуру
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0.0
        self.radius = 18

        self.setStyleSheet("""
        QLineEdit{
            background: transparent;
            color: #FFFFFF;
            font-size: 15px;
            font-weight: 600;
            border-radius:18px;
            padding:8px 12px;
            border: 2px solid transparent;
            selection-background-color: rgba(190,120,255,0.85);
            selection-color: #FFFFFF;
        }
        QLineEdit::placeholder{
            color: rgba(245,245,245,0.80);
        }
        """)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    def _tick(self):
        self.phase = (self.phase + 2) % 360
        self.update()

    def paintEvent(self, event):
        # сначала фон и рамка
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        center_qpoint = rect.center()
        center = QPointF(float(center_qpoint.x()),
                         float(center_qpoint.y()))

        grad = QConicalGradient(center, float(self.phase))
        grad.setColorAt(0.0, QColor(120, 255, 200, 150))
        grad.setColorAt(0.25, QColor(120, 180, 255, 150))
        grad.setColorAt(0.5, QColor(200, 160, 255, 150))
        grad.setColorAt(0.75, QColor(255, 200, 140, 150))
        grad.setColorAt(1.0, QColor(120, 255, 200, 150))

        inner = QRectF(rect).adjusted(1, 1, -1, -1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 8, 12, 240))
        p.drawRoundedRect(inner, self.radius - 1, self.radius - 1)

        pen = QPen(QBrush(grad), 2.0)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(rect), self.radius, self.radius)

        if self.hasSelectedText():
            glow_pen = QPen(QColor(255, 255, 255, 200), 3.0)
            p.setPen(glow_pen)
            outer = rect.adjusted(-1, -1, 1, 1)
            p.drawRoundedRect(QRectF(outer), self.radius + 1, self.radius + 1)

        p.end()

        # потом текст/каретка
        super().paintEvent(event)


# ========= Главное окно =========
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        logo_path = os.path.join(BASE_DIR, "logo.png")
        self.setWindowTitle("Юридический помощник РФ — офлайн")
        self.setWindowIcon(QIcon(logo_path))

        self.resize(1500, 900)

        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#000000"))
        self.setPalette(pal)

        # есть ли законы
        laws_dir = os.path.join(BASE_DIR, "laws")
        self.laws_ok = os.path.exists(laws_dir) and any(
            os.path.exists(os.path.join(laws_dir, f)) for f in LAWS_FILES
        )

        root_widget = QWidget()
        self.setCentralWidget(root_widget)

        layout = QHBoxLayout(root_widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # ===== LEFT =====
        left = QVBoxLayout()
        left.setSpacing(15)

        left_logo = RainbowAvatar(logo_path, 72)
        left.addWidget(left_logo)

        btn_home = Btn("Главная")
        btn_home.clicked.connect(self.home)

        btn_full = Btn("Полный экран")
        btn_exit = Btn("Выход")
        btn_exit.clicked.connect(self.close)
        btn_full.clicked.connect(self.toggle_full)

        left.addWidget(btn_home)
        left.addStretch()
        left.addWidget(btn_full)
        left.addWidget(btn_exit)

        # ===== CENTER =====
        center = Glass()
        c = QVBoxLayout(center)
        c.setSpacing(12)

        title = QLabel("Юридический помощник РФ")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            color:white;
            font-size:22px;
            font-weight:800;
            padding:10px;
            background: transparent;
        """)
        c.addWidget(title)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setStyleSheet("""
            color:white;
            background: transparent;
            border: none;
        """)
        c.addWidget(self.out, 1)

        self.input = RainbowLineEdit()
        self.input.setPlaceholderText("Например: за что могут проверять документы у гражданина…")
        c.addWidget(self.input)

        # Enter = отправить в ИИ
        self.input.returnPressed.connect(self.ask_ai)

        row = QHBoxLayout()
        self.btn_search = Btn("Найти закон")
        self.btn_ai = Btn("Объяснить ИИ")
        row.addWidget(self.btn_search)
        row.addWidget(self.btn_ai)
        c.addLayout(row)

        self.btn_search.clicked.connect(self.do_search)
        self.btn_ai.clicked.connect(self.ask_ai)

        # ===== RIGHT =====
        right = Glass()
        r = QVBoxLayout(right)
        r.setSpacing(12)

        info_text = (
            "Работает офлайн.\n"
            "• поиск по законам, написанным в txt файлах\n"
            "• ИИ локальный\n\n"
            "Ничего никуда не отправляется."
        )
        if not self.laws_ok:
            info_text += (
                "\n\n⚠ Не найдены txt-файлы законов в папке 'laws'.\n"
                "Папка должна лежать рядом с программой (app.py или .exe)."
            )

        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet("""
            color:white;
            font-size:13px;
            background: rgba(0,0,0,0.55);
            border-radius:18px;
            padding:10px;
        """)
        r.addWidget(info)
        r.addStretch()

        layout.addLayout(left, 1)
        layout.addWidget(center, 3)
        layout.addWidget(right, 2)

        self.is_full = False
        self.home()

    # ===== Логика =====
    def toggle_full(self):
        if self.is_full:
            self.showNormal()
        else:
            self.showFullScreen()
        self.is_full = not self.is_full

    def home(self):
        base = (
            "⚖ Юридический помощник РФ\n\n"
            "Важно понимать:\n\n"
            "• это не юридическая консультация\n"
            "• программа — ИИ-помощник\n"
            "• в базе не все законы РФ\n"
            "• тексты могут быть частично устаревшими\n"
            "• перед серьёзными действиями сверяйтесь с официальными источниками\n\n"
            "Используя приложение, вы соглашаетесь,\n"
            "что ответственность за решения несёте сами.\n"
        )
        if not self.laws_ok:
            base += (
                "\nДополнительно: пока не найдены файлы законов в папке 'laws'.\n"
                "Папка должна находиться рядом с программой (app.py или .exe)."
            )
        self.out.setText(base)

    def do_search(self):
        q = self.input.text().strip()
        if not q:
            return

        fname, frag = find_article(q, self.laws_ok)
        if not fname:
            if not self.laws_ok:
                self.out.append("\nПоиск: не найдены файлы законов в папке 'laws'.\n")
            else:
                self.out.append("\nНичего не найдено в загруженных txt-файлах.\n")
        else:
            self.out.append(f"\nНайдено в {fname}:\n{frag}\n")

    def ask_ai(self):
        q = self.input.text().strip()
        if not q:
            return

        fname, frag = find_article(q, self.laws_ok)
        frag = frag or ""

        self.out.append("\nИИ думает...\n")

        worker = AIWorker(q, frag)
        worker.finished.connect(self.ai_done)

        thread = Thread(target=worker.run, daemon=True)
        thread.start()

    def ai_done(self, text: str):
        self.out.append("Ответ ИИ:\n" + text + "\n")


def main():
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
