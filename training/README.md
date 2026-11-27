# Как обучить свою модель распознавания экрана Genshin Impact

Встроенная эвристика (`--backend heuristic`) читает HUD-полоски без обучения,
но боевые объекты (враги, сундуки, цифры урона) надёжно находит только YOLO.
Ниже — полный цикл: запись → разметка → обучение → запуск.

## 1. Запись геймплея

```bash
python scripts/record_gameplay.py --seconds 180 --monitor 1 \
    --out data/raw/fight1.mp4 --sample-dir data/frames --sample-every 1.0
```

Советы:
- снимайте **разные** сцены: бой, открытый мир, меню, ночь/день, разные отряды;
- разрешение фиксируйте одно (рекомендуется 1280×720 или 1920×1080);
- 300–800 кадров достаточно для первого рабочего варианта, 2000+ — для продакшена.

## 2. Разметка

Откройте `data/frames/images` в [CVAT](https://www.cvat.ai/), [Roboflow](https://roboflow.com)
или labelImg. Классы — из `training/dataset.yaml`:

| id | класс | что размечать |
|----|-------|---------------|
| 0 | player | играбельный персонаж |
| 1 | enemy | обычный противник |
| 2 | elite | элитный враг |
| 3 | boss | босс |
| 4 | enemy_hp_bar | белая полоска HP над врагом |
| 5 | player_hp_bar | зелёная полоска HP в HUD |
| 6 | stamina_bar | полоска стамины |
| 7 | damage_number | цифры урона |
| 8 | pickup | предметы на земле |
| 9 | chest | сундуки |
| 10 | dialogue | диалоговые окна |
| 11 | minimap | миникарта |

Минимум для боевого трекинга: **enemy, elite, boss, enemy_hp_bar, player_hp_bar**.
Экспортируйте в формате YOLO (txt на кадр) в `data/labels/{train,val}`.

## 3. Обучение

```bash
pip install -r requirements-ml.txt
python training/train_yolo.py --data training/dataset.yaml \
    --model yolov8n.pt --epochs 80 --imgsz 960 --device 0
```

- `yolov8n` — быстро (для слабых GPU), `yolov8s/m` — точнее;
- `--imgsz 960` хорошо ловит мелкие объекты (полоски HP, цифры урона);
- лучшие веса появятся в `runs/detect/genshin/weights/best.pt`.

## 4. Ускорение инференса (реальное время)

```bash
python training/export_onnx.py --weights runs/detect/genshin/weights/best.pt --half
```

Рекомендации для 60 FPS:
- `yolov8n` + FP16/TensorRT, `--imgsz 640–960`;
- захват только игровой области (`--region x,y,w,h`), а не всего экрана;
- троттлинг: детекция каждый кадр, HUD-чтение — каждый кадр (оно дешёвое).

## 5. Запуск

```bash
python scripts/run_live.py --backend yolo \
    --weights runs/detect/genshin/weights/best.pt --calibration config/my_calib.json
```

Не забудьте откалибровать HUD под свой интерфейс:

```bash
python -m genshinvision.calibrate screenshot.png -o config/my_calib.json
```
