# Какие веса и материалы нужно собрать

Материалы проверены 15 ноября 2026: все семь файлов `D53_legacy_weights.zip` совпадают с манифестом. В `D53_jury_run.zip` найден `final_d53_cpu_inputs.zip`, его SHA-256 совпадает с эталоном; `d41/embeddings_cuda.npz` присутствует. Старый `final_d53_inputs.zip` не заменяет CPU-архив. Архивы хранятся отдельно от Git.

Этот список относится к D53. Не всё нужно для каждого режима.

## Минимум для точной CPU-сборки

1. Полный официальный пакет Case2: CSV, служебные `.py`, изображения исторических aux. Для будущего инференса и обучения также нужны все кадры и их контекст. Путь передаётся до папки, содержащей `sample_submission.csv`.
2. **`final_d53_cpu_inputs.zip`** — зафиксированные входы прежних моделей, детекторные кэши, DINO-признаки и аудит D13. SHA: `cf46063f075c21814bd50499dd5d92845e2c829ee7abe85daf9dfffd5e55ee5e`.
3. Исходники `Counts_ruins` в коммите **`00b6fb3188f72ddebe1e297cf0e3ea6e0c5c67e2`**, включая train-аннотации `dima/full_review/full_train_annotations.json`.
4. Python3.10 и [зафиксированные зависимости](requirements/cached.txt).

Тяжёлые нейросетевые веса в этом режиме не требуются. Финальный D53 CSV и промежуточные D4/D21/D42/D48 восстанавливаются, а не должны уже лежать в архиве.

Архив CPU-входов должен содержать:

```text
SAMIR/submission_student_nb_pure.csv
SAMIR/submission_dfine_clean.csv
night-followup/submission_night_best.csv
submission-exp016/submission_exp016.csv
submission-exp016/yoloe/detections.jsonl
SAMIR/dfine/tables.csv
SAMIR/dfine/context.csv
SAMIR/student_pred/tables.csv
SAMIR/student_pred/context.csv
d6/instances.csv
d7/embeddings.npz
d13/annotations.json
d13/cases.json
d21/embeddings.npz
d41/embeddings.npz
d42/embeddings.npz
d41/embeddings_cuda.npz
manifest.json
```

Это список из исходного упаковщика. Имена и структура важны; содержимое не переносим в Git.

## Веса для инференса и обучения

Получен и проверен **`D53_legacy_weights.zip`**. SHA архива: `1b67a31aafe830d52f828269e7111828ea7574c3021e534092566cf2a35b76b9`.

| Файл | Размер, MB | Для чего |
|---|---:|---|
| `exp044/last.pth` |499.7| Точный обученный ученик. Нужен весь checkpoint; инференс берёт `ema.module`. |
| `dfine/dfine_x_obj365.pth` |254.5| Teacher/дополнительные предложения предметов и псевдометки. |
| `dfine/dfine_l_obj365_e25.pth` |128.4| Начальная инициализация при обучении ученика. |
| `yoloe/yoloe-26m-seg.pt` |70.4| Детектор YOLOE. |
| `yoloe/mobileclip2_b.ts` |253.8| Текстовый энкодер YOLOE; одного `.pt` детектора недостаточно. |
| `dinov2/model.safetensors` |88.3| DINOv2 Small. |
| `dinov2/config.json` |<0.001| Конфигурация DINOv2. |

Итого примерно1.30GB без учёта сжатия архивов и дополнительных файлов. Побайтовые размеры и SHA всех семи файлов — в [weights.json](manifests/weights.json); настройки YOLOE и версия энкодера — в [yoloe.json](manifests/yoloe.json).

## Материалы для обучения

- Полный пакет организаторов: `frames.csv`, `targets.csv`, `tables.csv`, `detections.csv`, `train_labels.csv`, `aux_frames.csv`, `aux_boxes.csv`, `sample_submission.csv`, изображения и контекстные кадры.
- Teacher-выходы `tables.csv`, `context.csv`, **`aux.csv`** из D-FINE-X. Последний нужен генератору псевдометок, хотя в CPU-архиве его нет. Их можно пересоздать после переноса teacher inference.
- Желательно сохранить готовую папку **`student_data/`** исходного EXP-044: `images/`, `train.json`, `val.json`, `stats.json`. Это позволит проверить состав5226 кропов и точность генерации. Это учебные данные, не файлы для Git.
- Исходный `student_cafe_l.yml`, весь набор включаемых D-FINE YAML и точная версия D-FINE checkout. Одного верхнего YAML недостаточно: часть оптимизатора, расписания и transforms наследуется.
- Исходные `pseudo_dataset.py`, `train_student.py`, журнал запуска, версии Python/PyTorch/CUDA, seed и логи обучения. Для точного продолжения обучения важны EMA и состояния optimizer/scheduler, если они есть в checkpoint.
- Ручная train-разметка и аудит: `full_train_annotations.json`, `d13/annotations.json`, `d13/cases.json`. В CPU-сборке часть берётся из закреплённого репозитория, часть — из архива.
- Для малых голов: их обучающие признаки/ID-сигнатуры, списки колонок и настройки scaler/порогов. Кэши D7/D21/D41/D42 находятся в CPU-архиве. Желательно также получить экспортированные Ridge EXP-010/016 и головы D14/D42/D48; если экспортов нет, их нужно пересоздать соответствующим рецептом. Имена таких экспортов пока не зафиксированы — не подменяем их выдуманными файлами.

## Для независимой проверки результата

Эталонный `submission_d53.csv` можно запросить отдельно для сравнения, но запускатель проверяет его SHA и без готового CSV. Нужны также исходный `score.py` организаторов и сохранённые метрики двух дат. Конкретные эталонные хэши промежуточных и финального CSV находятся в [reference.json](manifests/reference.json).

Наличие имени и хэша в манифесте не означает, что архив уже скачан или опубликован в новом репозитории. Данные, веса, кэши и выходные CSV хранятся локально; Git содержит их описание и контрольные суммы.
