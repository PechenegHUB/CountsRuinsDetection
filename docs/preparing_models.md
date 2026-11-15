# Подготовка проверяющих моделей

Обычный запуск `run.py` загружает три готовые модели из `models/heads.json`. В файле сохранены коэффициенты, свободные члены и пороги. Предсказание не вызывает обучение и не читает `train_labels.csv` или ручные аннотации.

Чтобы подготовить модели заново, нужны полный пакет организаторов, архив входов и `assets/train_annotations.json` из переданного командой пакета исходников. Разместите аннотации локально, например в `artifacts/training/train_annotations.json`.

Из корня проекта:

```powershell
.venv\Scripts\python.exe prepare_heads.py --data data/organizer --bundle artifacts/bundles/final_d53_cpu_inputs.zip --annotations artifacts/training/train_annotations.json --work artifacts/training/run --output artifacts/training/heads.json
```

Команда один раз обучает проверку чашек и две проверки слабых предметов, затем сохраняет коэффициенты и пороги. Выберите новую рабочую папку и новый выходной файл.

Для запуска с подготовленными моделями:

```powershell
.venv\Scripts\python.exe run.py --data data/organizer --models artifacts/training/heads.json --work artifacts/result_new
```

Эти головы используют признаки из архива. Базовые счётчики и исходная оценка занятости входят в сохранённые выходы предыдущих моделей; подготовка трёх голов их не переобучает.
