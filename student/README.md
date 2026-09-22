# Студенческая посылка Mars Rover

Архив содержит стартовое решение и **собственную собираемую копию среды** с
текущими 20 тренировочными биомами. При сборке Docker-образа пакет из архива
устанавливается поверх версии среды из базового образа.

## Что менять

- `train.py` — алгоритм, бюджет, гиперпараметры и сохранение модели;
- `model.py` — архитектуру агента и память;
- `python/mars_rover_env/configs/env.yaml` — генерацию трассы, сложность,
  завершение эпизода и reward для тренировок;
- `cpp/include/mars/custom_biomes.inc.hpp` — дополнительные тренировочные
  биомы. В файле есть отключённый рабочий пример;
- `cpp/include/mars/biome_bank.hpp`, `cpp/src/terrain.cpp` и
  `cpp/src/mechanics.cpp` — если нужна более глубокая модификация среды.

После изменения C++-части достаточно пересобрать посылку: Dockerfile сам
перекомпилирует расширение `_mars_rover_cpp`.

## Проверка перед отправкой

Из корня посылки соберите тот же образ, который будет запускать обучение:

```bash
docker build --tag mars-rover-submission-check .
```

Команда требует, чтобы образ `arena-base` был доступен локальному Docker.
Во время сборки автоматически запускается smoke-test. Он проверяет импорт
среды и платформенного entrypoint, контракт наблюдений и действий, фиксированную
конструкцию ровера, короткий прогон симуляции и формы выходов recurrent policy.
Если тест не прошёл, такой архив отправлять не следует.

## Фиксированный контракт ровера

Не изменяйте геометрию, органы управления и физику ровера. В частности, к
фиксированному контракту относятся:

- `cpp/include/mars/rover_rig.hpp` и `cpp/src/rover_rig.cpp`;
- `cpp/include/mars/action.hpp`;
- `cpp/include/mars/observation.hpp`;
- `cpp/include/mars/physics.hpp` и `cpp/src/physics.cpp`;
- `python/mars_rover_env/configs/rover_rig.yaml`;
- публичный порядок `ACTION_MACROS` в `python/mars_rover_env/actions.py`.

Тестовая среда использует исходный контракт ровера, поэтому обучение на его
модифицированной версии даст несопоставимую модель.

## Добавление биома

1. Скопируйте `ExampleTrainingBiome` в
   `cpp/include/mars/custom_biomes.inc.hpp` и задайте уникальные `id()` и
   параметры `sample_params()`.
2. Добавьте статический экземпляр в `custom_biomes::append`.
3. Убедитесь, что `split()` возвращает `BiomeSplit::Train`.
4. Локально соберите среду и проверьте каталог:

```bash
python -m pip install --no-build-isolation --no-deps --force-reinstall .
python -c "import _mars_rover_cpp as m; print(m.biome_catalog())"
```

## Формат посылки

Загружайте ZIP без дополнительной внешней папки. В его корне должны оставаться
`Dockerfile`, `README.md`, `train.py`, `model.py`, `pyproject.toml`, `setup.py`,
а также каталоги `cpp/` и `python/`.

Во время обучения программа должна создать `/output/policy.onnx`. Стартовый
вариант использует `vendor.meta_ppo`, но алгоритм можно заменить целиком, если
сохранить ONNX-контракт платформы.

Ресурсы стартового сервера: 6 CPU, 24 ГиБ RAM, половина RTX 5090, 15 ГиБ VRAM,
2 ГиБ `/tmp`, без доступа к интернету и с лимитом около двух часов.
