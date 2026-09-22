# Organizer GUI

Этот GUI запускается только с организаторской сборкой среды. Он сохраняет полный
каталог, закрытые визуальные эффекты и расширенную телеметрию и не должен входить
в студенческий архив.

```bash
cd organizer
make build
make play ARGS="--list-biomes"
make play ARGS="--config python/mars_rover_env/configs/play.yaml --debug"
```

Проверка границы между поставками:

```bash
python gui/test_student_boundary.py
```
