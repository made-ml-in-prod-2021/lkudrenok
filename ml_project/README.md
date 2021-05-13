Обучение модели:

<code>python run_pipeline.py train configs/train_config_base.yml</code><br>
<code>python run_pipeline.py train configs/train_config_advanced.yml</code>

Предсказание по обученной модели:

<code>python run_pipeline.py predict configs/predict_config.yml</code>

Запуск тестов:

<code>pytest</code>

Запуск линтера:

<code>pylint run_pipeline.py ml_project</code>

Самооценка:

- [x] назвать ветку homework1 ***(max/fact: +1)***
- [x] положить код в папку ml_project
- [x] в пулл-реквесте описать, что сделано ***(max/fact: +2)***
- [x] EDA в папке notebooks ***(max: +3, fact: +1)***
- [x] модульная структура ***(max/fact: +2)***
- [x] логирование ***(max/fact: +2)***
- [x] тестирование модулей и всего пайплайна ***(max/fact: +3)***
- [x] для тестов генерируются синтетические данные ***(max/fact: +3)***
- [x] два конфига для обучения ***(max/fact: +3)***
- [x] использование dataclass ***(max/fact: +3)***
- [x] написан и протестирован кастомный трансформер ***(max/fact: +3)***
- [x] обучить модель, описать в readme ***(max/fact: +3)***
- [x] predict на основе артефактов, описать в readme ***(max/fact: +3)***
- [ ] EXTRA: использовать hydra ***(max: +3, fact: +0)***
- [x] EXTRA: CI (tests, linter) на основе github actions ***(max/fact: +3)***
- [x] EXTRA: самооценка ***(max/fact: +1)***

Итого по самооценке: 33 балла.
