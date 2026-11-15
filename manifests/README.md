# Манифесты

- source.json — точный старый checkout и entrypoint.
- reference.json — SHA CPU-архива и промежуточных/финального CSV.
- bundles.json — отдельно передаваемые архивы.
- weights.json — точные пути, размеры и SHA checkpoint.
- yoloe.json — происхождение YOLOE/энкодера и inference settings.

Хэши скопированы из закреплённого источника, а не получены проверкой отсутствующих здесь архивов. Проверить фактически полученные файлы можно CLI check-assets.
