# GenshinVision — ИИ-модель «что и где» на экране Genshin Impact

Распознавание экрана в **реальном времени**: где противник и куда он движется,
сколько HP у вас и у врагов, стамина, босс, готовность ультимейтов — и всё это
в виде JSON-состояния 20–60 раз в секунду с живым оверлеем в браузере.

```
┌─────────────┐   ┌──────────────────────┐   ┌───────────────────┐
│ Захват кадра│ → │ Детектор             │ → │ HUD-анализ        │
│ (mss/видео) │   │ YOLOv8 / эвристика   │   │ HP, стамина, босс │
└─────────────┘   └──────────────────────┘   └───────────────────┘
                          │                           │
                          ▼                           ▼
                  ┌───────────────────┐      ┌─────────────────┐
                  │ Трекер (Kalman +  │ ←──  │ Стейт-билдер    │
                  │ Hungarian, SORT)  │      │ JSON 20-60 Гц   │
                  └───────────────────┘      └─────────────────┘
                          │ скорость, прогноз пути (0.5 с)
                          ▼
              WebSocket + веб-оверлей / бот / лог
```

## Что умеет

| Сигнал | Как получается |
|--------|----------------|
| **HP игрока** (текущее/макс., доля) | цветовой анализ полоски + опционально OCR текста `HP 4812/12000` |
| **HP каждого врага** | белая полоска над врагом → доля заполнения |
| **Позиция врагов** | YOLO-детекция (или эвристика по полоскам HP) |
| **Куда движется противник** | Kalman-фильтр по каждому треку → вектор скорости px/с + предсказанная позиция через 0.5 с |
| **Стамина, полоска босса** | цветовой анализ калиброванных ROI |
| **Партия: HP, готовность ульт** | ROI слева/справа + порог яркости иконок |
| **«В бою / спокойно»** | есть активные треки врагов или полоска босса |
| **Живой оверлей** | браузер: кадр + боксы + стрелки скорости + панель состояния (WebSocket, ~20 Гц) |

Выход — единый JSON на каждый кадр (см. «Формат состояния» ниже). Его можно
пить что угодно: оверлей, бота, телеметрию, тренажёр реакции.

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # базовый конвейер (демо, эвристика, стрим)
pip install -r requirements-ml.txt       # + YOLO (torch, ultralytics) — для реального распознавания
```

## Быстрый старт

### 1. Демо без игры (синтетическая сцена)

```bash
python scripts/run_demo.py --port 8000
# → откройте http://localhost:8000
```

Веб-страница показывает кадр, боксы врагов, стрелки скорости, прогноз позиции,
HP/стамину/босса. Это же — проверка, что всё работает без установки игры.

### 2. Реальная игра (захват экрана)

```bash
# сначала откалибруйте HUD под свой интерфейс:
python -m genshinvision.calibrate screenshot.png -o config/my_calib.json

# эвристика (без обучения, читает HUD-полоски):
python scripts/run_live.py --calibration config/my_calib.json

# полная YOLO-модель (враги, предметы, цифры урона):
python scripts/run_live.py --backend yolo --weights runs/detect/genshin/weights/best.pt \
    --calibration config/my_calib.json --monitor 1
```

Полезные флаги: `--region x,y,w,h` (захват только окна игры — быстрее),
`--display` (доп. окно OpenCV), `--device 0` (GPU), `--fps 60`.

### 3. Оффлайн по видео

```bash
python scripts/run_video.py --input fight.mp4 \
    --output out/annotated.mp4 --states out/states.jsonl
```

## Обучение своей модели (YOLOv8)

Эвристика хорошо читает HUD, но врагов, сундуки и цифры урона надёжно находит
обученная модель. Полный цикл — запись геймплея → разметка → дообучение →
экспорт в ONNX/TensorRT — описан в [training/README.md](training/README.md):

```bash
python scripts/record_gameplay.py --seconds 180 --out data/raw/fight1.mp4 --sample-dir data/frames
# разметьте кадры (CVAT/Roboflow), затем:
python training/train_yolo.py --data training/dataset.yaml --model yolov8n.pt --epochs 80 --device 0
python training/export_onnx.py --weights runs/detect/genshin/weights/best.pt --half
```

12 классов: `player, enemy, elite, boss, enemy_hp_bar, player_hp_bar,
stamina_bar, damage_number, pickup, chest, dialogue, minimap`.

## Формат состояния (WebSocket `/ws`, GET `/state`)

```json
{
  "timestamp": 1.5333,
  "frame_size": [1280, 720],
  "fps": 30.0,
  "player": {"hp_frac": 0.776, "hp_cur": 9307, "hp_max": 12000, "stamina_frac": 0.827},
  "enemies": [
    {
      "id": 2, "class": "enemy",
      "bbox": [548.6, 189.1, 691.1, 398.1],
      "hp_frac": 0.5474,
      "velocity_px_s": [42.3, -31.5],
      "speed_px_s": 52.8,
      "predicted_bbox": [569.8, 173.3, 712.3, 382.3],
      "age_s": 1.5, "hits": 46
    }
  ],
  "ui": {"in_combat": true, "boss_hp_frac": 0.885,
         "party_hp_fracs": [0.91, 0.68, 0.77], "burst_ready": [false, true, true]},
  "backend": "heuristic"
}
```

- `velocity_px_s` — куда идёт противник, в пикселях экрана в секунду;
- `predicted_bbox` — где он окажется через 0.5 с при текущем курсе;
- `hp_frac` — доля HP (0..1); `hp_cur/hp_max` игрока — из OCR или `hp_max` в калибровке.

## Калибровка HUD

Все HUD-регионы заданы в **долях** ширины/высоты экрана, поэтому одна
калибровка работает на любом разрешении:

```bash
python -m genshinvision.calibrate screenshot.png -o config/my_calib.json
# ЛКМ — обвести регион, Enter — принять, s — пропустить, q — выход
```

Регионы: `player_hp_bar`, `player_hp_text`, `stamina_bar`, `boss_hp_bar`,
`party_bars`, `burst_icons`, `minimap`, `combat_region`. Дефолты —
[config/default_calibration.json](config/default_calibration.json).

## Производительность (реальное время)

- захват только окна игры (`--region`) вместо всего экрана;
- `yolov8n` + FP16/TensorRT (`training/export_onnx.py`) → 60+ FPS на современном GPU;
- `--imgsz 640` для скорости / `960` для мелких объектов;
- эвристика вообще не требует GPU: ~30 FPS на CPU при 720p.

## Структура проекта

```
genshinvision/
  capture.py     # захват экрана / видео / картинок
  detect.py      # детекторы: HeuristicDetector, YoloDetector
  hud.py         # чтение HP/стамины/босса по цвету (калиброванные ROI)
  track.py       # SORT-трекер: Kalman + Hungarian, скорость и прогноз
  state.py       # сборка GameState (JSON)
  pipeline.py    # склейка + аннотированный кадр
  stream.py      # FastAPI + WebSocket + хаб
  synthetic.py   # синтетическая сцена с ground truth (демо/тесты)
  calibrate.py   # интерактивная калибровка HUD
scripts/         # run_demo / run_live / run_video / record_gameplay
overlay/         # веб-оверлей (HTML+JS)
training/        # dataset.yaml, train_yolo.py, export_onnx.py, инструкция
tests/           # pytest: трекер, HUD, сквозной прогон
```

Проверка: `python -m pytest tests/ -q` (10 тестов, включая сверку с ground truth:
HP ±3%, позиция врагов <60 px, скорость <70 px/с).

## Ограничения и юридическая часть

- Это **пассивное зрение**: только чтение пикселей экрана. Никакого чтения
  памяти игры, инжекта, автоматических нажатий — это не нарушает механику
  античита, но использование оверлеев/помощников может противоречить правилам
  игры — используйте для анализа, обучения и стриминга на свой риск.
- Эвристика — рабочий фолбэк и основа для демо; для боёв с частицами/эффектами
  ставьте обученную YOLO-модель (`training/README.md`).
- OCR HP-текста опционален (`pip install pytesseract` + tesseract-ocr); без
  него абсолютные значения HP берутся из `hp_max` в калибровке.
