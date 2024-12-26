Callbacks
=========

В этой секции представлены различные классы колбеков, используемые в библиотеке **Diffusion**. Колбеки позволяют расширять функциональность тренировочного процесса, добавляя дополнительные шаги или изменяя поведение модели во время обучения.

.. toctree::
   :maxdepth: 2
   :caption: Содержание

   Callback
   TrainCB
   DeviceCB
   AccelerateCB
   HooksCallback
   ActivationStats
   BaseSchedCB
   BatchSchedCB
   BatchTransformCB
   CapturePredsCB
   CompletionCB
   DDPMCB
   DDPMCB2
   EpochSchedCB
   HasLearnCB
   ImageLogCB
   ImageOptCB
   LRFinderCB
   MetricsCB
   MixedPrecisionCB
   MultiGPUsCallback
   ProgressCB
   RecorderCB
   SingleBatchCB
   with_cbs
Callback
--------

Класс :class:`~callbacks.Callback` является базовым классом для всех колбеков в библиотеке **Diffusion**. Он определяет общую структуру и порядок выполнения колбеков во время тренировочного цикла.

**Основные атрибуты:**

- **order**: Определяет порядок выполнения колбека относительно других колбеков. По умолчанию значение `0`.

**Основные методы:**

- **`before_fit(self, learn)`**: Вызывается перед началом обучения.
- **`after_fit(self, learn)`**: Вызывается после завершения обучения.
- **`before_batch(self, learn)`**: Вызывается перед обработкой каждого батча.
- **`after_batch(self, learn)`**: Вызывается после обработки каждого батча.
- **`predict(self, learn)`**: Выполняет предсказание модели.
- **`get_loss(self, learn)`**: Вычисляет функцию потерь.
- **`backward(self, learn)`**: Выполняет обратное распространение ошибки.
- **`step(self, learn)`**: Обновляет параметры оптимизатора.
- **`zero_grad(self, learn)`**: Обнуляет градиенты оптимизатора перед новым шагом обучения.

**Пример использования:**

.. code-block:: python

    from callbacks import Callback

    class MyCustomCB(Callback):
        def before_fit(self, learn):
            print("Начало обучения")

        def after_fit(self, learn):
            print("Обучение завершено")

---

TrainCB
-------

Класс :class:`~callbacks.TrainCB` наследуется от базового класса :class:`~callbacks.Callback` и предназначен для управления основными этапами тренировочного цикла, такими как предсказание, вычисление потерь, обратное распространение и обновление оптимизатора.

**Основные атрибуты:**

- **n_inp**: Количество входных данных, используемых моделью для предсказания. По умолчанию значение `1`.

**Основные методы:**

- **`__init__(self, n_inp=1)`**: Инициализирует колбек с заданным количеством входных данных.
- **`predict(self, learn)`**: Выполняет предсказание модели на основе текущего батча данных.
- **`get_loss(self, learn)`**: Вычисляет функцию потерь между предсказаниями модели и истинными значениями.
- **`backward(self, learn)`**: Выполняет обратное распространение ошибки для вычисленной функции потерь.
- **`step(self, learn)`**: Обновляет параметры оптимизатора на основе вычисленных градиентов.
- **`zero_grad(self, learn)`**: Обнуляет градиенты оптимизатора перед новым шагом обучения.

**Пример использования:**

.. code-block:: python

    from callbacks import TrainCB

    class CustomTrainCB(TrainCB):
        def get_loss(self, learn):
            # Пользовательская функция потерь
            learn.loss = custom_loss_function(learn.preds, *learn.batch[self.n_inp:])

---

DeviceCB
--------

Класс :class:`~callbacks.DeviceCB` наследуется от базового класса :class:`~callbacks.Callback` и отвечает за перенос модели и данных на указанное устройство (CPU или GPU). Это обеспечивает эффективное использование вычислительных ресурсов и ускоряет процесс обучения.

**Основные атрибуты:**

- **device**: Устройство, на которое будет перенесена модель и данные. Возможные значения: `'cpu'`, `'cuda'` и другие поддерживаемые устройствами.

**Основные методы:**

- **`__init__(self, device='cpu')`**: Инициализирует колбек с указанным устройством.
- **`before_fit(self, learn)`**: Переносит модель на указанное устройство перед началом обучения.
- **`before_batch(self, learn)`**: Переносит текущий батч данных на указанное устройство перед обработкой.

**Пример использования:**

.. code-block:: python

    from callbacks import DeviceCB

    device_cb = DeviceCB(device='cuda')
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[device_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )

---

AccelerateCB
------------

Класс :class:`~callbacks.AccelerateCB` наследуется от :class:`~callbacks.TrainCB` и интегрируется с библиотекой [Accelerate](https://github.com/huggingface/accelerate) для упрощения процесса обучения на различных устройствах с поддержкой смешанной точности.

**Основные атрибуты:**

- **order**: Определяет порядок выполнения колбека относительно других колбеков. Устанавливается как `DeviceCB.order + 10` для гарантии выполнения после `DeviceCB`.

**Основные методы:**

- **`__init__(self, n_inp=1, mixed_precision="fp16")`**: Инициализирует колбек с заданным количеством входных данных и типом смешанной точности.
- **`before_fit(self, learn)`**: Подготавливает модель, оптимизатор и загрузчики данных с помощью `Accelerator` перед началом обучения.
- **`backward(self, learn)`**: Выполняет обратное распространение ошибки с использованием `Accelerator`.

**Пример использования:**

.. code-block:: python

    from callbacks import AccelerateCB

    accelerate_cb = AccelerateCB(n_inp=1, mixed_precision="fp16")
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[accelerate_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

HooksCallback
-------------

Класс :class:`~callbacks.HooksCallback` наследуется от :class:`~callbacks.Callback` и предназначен для добавления пользовательских хуков к модулям модели. Хуки позволяют выполнять дополнительные действия во время тренировочного процесса, такие как мониторинг активаций или изменение поведения модели.

**Основные атрибуты:**

- **hookfunc**: Функция, которая будет вызываться при срабатывании хука.
- **mod_filter**: Фильтр модулей, к которым будут применяться хуки.
- **on_train**: Флаг, определяющий, будет ли хук вызываться во время тренировки.
- **on_valid**: Флаг, определяющий, будет ли хук вызываться во время валидации.
- **mods**: Список модулей, к которым будут применяться хуки. Если не задан, используется `mod_filter`.

**Основные методы:**

- **`__init__(self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None)`**: Инициализирует колбек с заданными параметрами.
- **`before_fit(self, learn)`**: Добавляет хуки к указанным модулям перед началом обучения.
- **`_hookfunc(self, learn, *args, **kwargs)`**: Внутренняя функция, вызываемая хуками.
- **`after_fit(self, learn)`**: Удаляет все добавленные хуки после завершения обучения.
- **`__iter__(self)`**: Позволяет итерироваться по хукам.
- **`__len__(self)`**: Возвращает количество хуков.

**Пример использования:**

.. code-block:: python

    from callbacks import HooksCallback
    import torch.nn as nn

    def my_hook(learn, module, input, output):
        print(f"Модуль: {module}, Вход: {input}, Выход: {output}")

    hooks_cb = HooksCallback(
        hookfunc=my_hook,
        mod_filter=lambda m: isinstance(m, nn.Linear),
        on_train=True,
        on_valid=True
    )

    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[hooks_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

ActivationStats
---------------

Класс :class:`~callbacks.ActivationStats` наследуется от :class:`~callbacks.HooksCallback` и предназначен для сбора и визуализации статистики активаций нейронных сетей, таких как гистограммы распределения активаций и графики минимальных значений.

**Примечание:** Класс `ActivationStats` может быть неэффективен по использованию памяти.

**Основные атрибуты:**

- Наследует все атрибуты от `HooksCallback`.

**Основные методы:**

- **`__init__(self, mod_filter=fc.noop)`**: Инициализирует колбек с заданным фильтром модулей.
- **`color_dim(self, figsize=(11, 5))`**: Визуализирует гистограммы распределения активаций.
- **`dead_chart(self, figsize=(11, 5))`**: Визуализирует графики минимальных значений активаций.
- **`plot_stats(self, figsize=(10, 4))`**: Визуализирует статистику средних значений и стандартных отклонений активаций.

**Пример использования:**

.. code-block:: python

    from callbacks import ActivationStats
    import torch.nn as nn

    activation_stats_cb = ActivationStats(mod_filter=lambda m: isinstance(m, nn.Conv2d))
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[activation_stats_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

    # Визуализация собранной статистики
    activation_stats_cb.color_dim()
    activation_stats_cb.dead_chart()
    activation_stats_cb.plot_stats()

---

BaseSchedCB
----------

Класс :class:`~callbacks.BaseSchedCB` наследуется от :class:`~callbacks.Callback` и предназначен для интеграции расписаний (schedulers) в тренировочный цикл. Он управляет шагами расписания на основе текущего состояния оптимизатора.

**Основные атрибуты:**

- **sched**: Расписание (scheduler), которое будет управлять оптимизатором.

**Основные методы:**

- **`__init__(self, sched)`**: Инициализирует колбек с заданным расписанием.
- **`before_fit(self, learn)`**: Подготавливает расписание перед началом обучения.
- **`step(self, learn)`**: Выполняет шаг расписания во время обучения.

**Пример использования:**

.. code-block:: python

    from callbacks import BaseSchedCB
    from torch.optim.lr_scheduler import StepLR

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    sched_cb = BaseSchedCB(sched=scheduler)

    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[sched_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=30)

---

BatchSchedCB
------------

Класс :class:`~callbacks.BatchSchedCB` наследуется от :class:`~callbacks.BaseSchedCB` и предназначен для выполнения шагов расписания после каждого батча во время обучения.

**Основные атрибуты:**

- Наследует все атрибуты от `BaseSchedCB`.

**Основные методы:**

- **`after_batch(self, learn)`**: Выполняет шаг расписания после обработки каждого батча, если обучение активно.

**Пример использования:**

.. code-block:: python

    from callbacks import BatchSchedCB
    from torch.optim.lr_scheduler import StepLR

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    batch_sched_cb = BatchSchedCB(sched=scheduler)

    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[batch_sched_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

BatchTransformCB
----------------

Класс :class:`~callbacks.BatchTransformCB` наследуется от :class:`~callbacks.Callback` и предназначен для применения пользовательских трансформаций к каждому батчу данных перед обработкой.

**Основные атрибуты:**

- **tfm**: Трансформация, которая будет применяться к батчу данных.
- **on_train**: Флаг, определяющий, будет ли трансформация применяться во время тренировки.
- **on_val**: Флаг, определяющий, будет ли трансформация применяться во время валидации.

**Основные методы:**

- **`__init__(self, tfm, on_train=True, on_val=True)`**: Инициализирует колбек с заданной трансформацией и флагами применения.
- **`before_batch(self, learn)`**: Применяет трансформацию к текущему батчу данных, если соответствующие флаги установлены.

**Пример использования:**

.. code-block:: python

    from callbacks import BatchTransformCB
    from torchvision import transforms

    # Определение трансформаций
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    batch_transform_cb = BatchTransformCB(tfm=data_transforms, on_train=True, on_val=False)

    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[batch_transform_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

CapturePredsCB
-------------

Класс :class:`~callbacks.CapturePredsCB` наследуется от :class:`~callbacks.Callback` и предназначен для захвата предсказаний модели и истинных значений (таргетов) во время обучения. Это полезно для последующего анализа или визуализации результатов.

**Основные атрибуты:**

- **all_preds**: Список для хранения всех предсказаний модели.
- **all_targs**: Список для хранения всех истинных значений.

**Основные методы:**

- **`before_fit(self, learn)`**: Инициализирует списки для хранения предсказаний и истинных значений перед началом обучения.
- **`after_batch(self, learn)`**: Добавляет предсказания и истинные значения текущего батча в соответствующие списки.

**Пример использования:**

.. code-block:: python

    from callbacks import CapturePredsCB

    capture_preds_cb = CapturePredsCB()
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[capture_preds_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

    # Получение всех предсказаний и истинных значений
    all_preds, all_targs = capture_preds_cb.all_preds, capture_preds_cb.all_targs

---

CompletionCB
------------

Класс :class:`~callbacks.CompletionCB` наследуется от :class:`~callbacks.Callback` и предназначен для отслеживания количества обработанных батчей и вывода сообщения о завершении обучения.

**Основные методы:**

- **`before_fit(self, learn)`**: Инициализирует счетчик `count` перед началом обучения.
- **`after_batch(self, learn)`**: Увеличивает счетчик `count` после обработки каждого батча.
- **`after_fit(self, learn)`**: Выводит сообщение о завершении обучения и количестве обработанных батчей.

**Пример использования:**

.. code-block:: python

    from callbacks import CompletionCB

    completion_cb = CompletionCB()
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[completion_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

DDPMCB
------

Класс :class:`~callbacks.DDPMCB` наследуется от :class:`~callbacks.Callback` и предназначен для реализации процесса шумования (noisify) и генерации выборок с использованием модели диффузии.

**Основные атрибуты:**

- **n_steps**: Количество шагов диффузии.
- **beta_min**: Минимальное значение бета.
- **beta_max**: Максимальное значение бета.
- **beta**: Линейно распределенные значения бета.
- **alpha**: 1 - beta.
- **alpha_bar**: Кумулятивное произведение альфа.
- **sigma**: Квадратные корни бета.

**Основные методы:**

- **`__init__(self, n_steps, beta_min, beta_max)`**: Инициализирует колбек с заданными параметрами.
- **`before_batch(self, learn)`**: Применяет шумование к батчу данных перед обучением.
- **`sample(self, model, sz)`**: Генерирует выборки с использованием модели и параметров диффузии.

**Пример использования:**

.. code-block:: python

    from callbacks import DDPMCB

    ddpm_cb = DDPMCB(n_steps=1000, beta_min=0.0001, beta_max=0.02)
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[ddpm_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

DDPMCB2
-------

Класс :class:`~callbacks.DDPMCB2` наследуется от :class:`~callbacks.Callback` и предназначен для модификации предсказаний модели после выполнения предсказания.

**Основные методы:**

- **`after_predict(self, learn)`**: Изменяет предсказания модели, вызывая метод `sample` на предсказаниях.

**Пример использования:**

.. code-block:: python

    from callbacks import DDPMCB2

    ddpm_cb2 = DDPMCB2()
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[ddpm_cb2],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

EpochSchedCB
------------

Класс :class:`~callbacks.EpochSchedCB` наследуется от :class:`~callbacks.BaseSchedCB` и предназначен для выполнения шагов расписания после каждой эпохи во время обучения.

**Основные методы:**

- **`after_epoch(self, learn)`**: Выполняет шаг расписания после завершения каждой эпохи.

**Пример использования:**

.. code-block:: python

    from callbacks import EpochSchedCB
    from torch.optim.lr_scheduler import StepLR

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    epoch_sched_cb = EpochSchedCB(sched=scheduler)

    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[epoch_sched_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=30)

---

HasLearnCB
----------

Класс :class:`~callbacks.HasLearnCB` наследуется от :class:`~callbacks.Callback` и предназначен для сохранения ссылки на объект `learn` во время обучения, что может быть полезно для доступа к нему в других колбеках.

**Основные методы:**

- **`before_fit(self, learn)`**: Сохраняет ссылку на объект `learn` перед началом обучения.
- **`after_fit(self, learn)`**: Удаляет ссылку на объект `learn` после завершения обучения.

**Пример использования:**

.. code-block:: python

    from callbacks import HasLearnCB

    has_learn_cb = HasLearnCB()
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[has_learn_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

    # Доступ к объекту learn
    some_function(has_learn_cb.learn)

---

ImageLogCB
----------

Класс :class:`~callbacks.ImageLogCB` наследуется от :class:`~callbacks.Callback` и предназначен для логирования изображений на определённых интервалах во время обучения.

**Основные атрибуты:**

- **log_every**: Интервал (в батчах) для логирования изображений.
- **images**: Список для хранения логированных изображений.
- **i**: Счетчик обработанных батчей.

**Основные методы:**

- **`__init__(self, log_every=10)`**: Инициализирует колбек с заданным интервалом логирования.
- **`after_batch(self, learn)`**: Логирует изображение, если достигнут заданный интервал.
- **`after_fit(self, learn)`**: Отображает все логированные изображения после завершения обучения.

**Пример использования:**

.. code-block:: python

    from callbacks import ImageLogCB

    image_log_cb = ImageLogCB(log_every=10)
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[image_log_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

ImageOptCB
----------

Класс :class:`~callbacks.ImageOptCB` наследуется от :class:`~callbacks.TrainCB` и предназначен для оптимизации предсказаний модели без дополнительных параметров.

**Основные методы:**

- **`predict(self, learn)`**: Выполняет предсказание модели без передачи дополнительных входных данных.
- **`get_loss(self, learn)`**: Вычисляет функцию потерь, используя предсказания модели.

**Пример использования:**

.. code-block:: python

    from callbacks import ImageOptCB

    image_opt_cb = ImageOptCB()
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[image_opt_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

LRFinderCB
---------

Класс :class:`~callbacks.LRFinderCB` наследуется от :class:`~callbacks.Callback` и предназначен для поиска оптимальной скорости обучения (learning rate) с использованием метода поиска скорости обучения.

**Основые атрибуты:**

- **gamma**: Коэффициент увеличения скорости обучения на каждом шаге.
- **max_mult**: Максимальный множитель для предотвращения слишком большого увеличения скорости обучения.
- **sched**: Расписание (scheduler) для управления скоростью обучения.
- **lrs**: Список для хранения скоростей обучения.
- **losses**: Список для хранения значений функции потерь.
- **min**: Минимальное значение функции потерь.

**Основые методы:**

- **`__init__(self, gamma=1.3, max_mult=3)`**: Инициализирует колбек с заданными параметрами.
- **`before_fit(self, learn)`**: Инициализирует расписание и списки для логирования перед началом обучения.
- **`after_batch(self, learn)`**: Логирует текущую скорость обучения и значение функции потерь, а также проверяет условия остановки поиска скорости обучения.
- **`cleanup_fit(self, learn)`**: Отображает график зависимости функции потерь от скорости обучения после завершения поиска.

**Пример использования:**

.. code-block:: python

    from callbacks import LRFinderCB

    lr_finder_cb = LRFinderCB(gamma=1.3, max_mult=3)
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[lr_finder_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

MetricsCB
---------

Класс :class:`~callbacks.MetricsCB` наследуется от :class:`~callbacks.Callback` и предназначен для вычисления и логирования различных метрик во время обучения и валидации.

**Основные атрибуты:**

- **metrics**: Словарь метрик, которые будут вычисляться.
- **all_metrics**: Копия словаря метрик для хранения накопленных значений.

**Основные методы:**

- **`__init__(self, *ms, **metrics)`**: Инициализирует колбек с заданными метриками.
- **`_log(self, d)`**: Выводит лог данных.
- **`before_fit(self, learn)`**: Устанавливает текущий объект метрик в объект `learn`.
- **`before_epoch(self, learn)`**: Сбрасывает значения всех метрик перед началом новой эпохи.
- **`after_epoch(self, learn)`**: Вычисляет и логирует средние значения метрик после завершения эпохи.
- **`after_batch(self, learn)`**: Обновляет значения метрик после обработки каждого батча.

**Пример использования:**

.. code-block:: python

    from callbacks import MetricsCB
    from torcheval.metrics import Mean

    metrics_cb = MetricsCB(accuracy=Mean())
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[metrics_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

MixedPrecisionCB
---------------

Класс :class:`~callbacks.MixedPrecisionCB` наследуется от :class:`~callbacks.TrainCB` и предназначен для использования смешанной точности (mixed precision) при обучении модели, что позволяет ускорить обучение и снизить потребление памяти.

**Основые атрибуты:**

- **order**: Определяет порядок выполнения колбека относительно других колбеков. Устанавливается как `DeviceCB.order + 10`.

**Основые методы:**

- **`__init__(self, n_inp=1, mixed_precision="fp16")`**: Инициализирует колбек с заданным количеством входных данных и типом смешанной точности.
- **`before_fit(self, learn)`**: Инициализирует `GradScaler` для масштабирования градиентов перед началом обучения.
- **`before_batch(self, learn)`**: Включает автоматическое смешанное точность перед обработкой батча.
- **`after_loss(self, learn)`**: Выключает автоматическое смешанное точность после вычисления потерь.
- **`backward(self, learn)`**: Выполняет обратное распространение ошибки с использованием `GradScaler`.
- **`step(self, learn)`**: Выполняет шаг оптимизатора с использованием `GradScaler` и обновляет масштабирование градиентов.

**Пример использования:**

.. code-block:: python

    from callbacks import MixedPrecisionCB

    mixed_precision_cb = MixedPrecisionCB(n_inp=1, mixed_precision="fp16")
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[mixed_precision_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

MultiGPUsCallback
-----------------

Класс :class:`~callbacks.MultiGPUsCallback` наследуется от :class:`~callbacks.DeviceCB` и предназначен для автоматического использования нескольких GPU при доступности, что позволяет ускорить процесс обучения моделей.

**Основые методы:**

- **`__init__(self)`**: Инициализирует колбек, определяя устройство (GPU или CPU).
- **`before_fit(self, learn)`**: Если доступно несколько GPU, преобразует модель в `nn.DataParallel` для использования нескольких устройств, затем вызывает метод `before_fit` базового класса.
- **`after_fit(self, learn)`**: Удаляет обертку `nn.DataParallel` после завершения обучения, возвращая модель к исходному состоянию.

**Пример использования:**

.. code-block:: python

    from callbacks import MultiGPUsCallback

    multi_gpus_cb = MultiGPUsCallback()
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[multi_gpus_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

ProgressCB
----------

Класс :class:`~callbacks.ProgressCB` наследуется от :class:`~callbacks.Callback` и предназначен для отображения прогресса обучения с помощью библиотеки `fastprogress`.

**Основые атрибуты:**

- **plot**: Флаг, определяющий, будет ли отображаться график потерь во время обучения.
- **mbar**: Объект `master_bar` для отображения прогресса обучения.
- **first**: Флаг, определяющий, выводился ли уже заголовок таблицы метрик.
- **losses**: Список для хранения значений потерь на тренировочных данных.
- **val_losses**: Список для хранения значений потерь на валидационных данных.

**Основые методы:**

- **`__init__(self, plot=False)`**: Инициализирует колбек с заданным флагом отображения графика.
- **`before_fit(self, learn)`**: Инициализирует `master_bar` и настраивает логирование метрик.
- **`_log(self, d)`**: Выводит лог данных в `master_bar`.
- **`before_epoch(self, learn)`**: Инициализирует прогресс-бар для текущей эпохи.
- **`after_batch(self, learn)`**: Обновляет комментарий прогресс-бара с текущим значением потерь и обновляет график, если включено.
- **`after_epoch(self, learn)`**: Обновляет график потерь после завершения эпохи, если включено.

**Пример использования:**

.. code-block:: python

    from callbacks import ProgressCB

    progress_cb = ProgressCB(plot=True)
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[progress_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

---

RecorderCB
---------

Класс :class:`~callbacks.RecorderCB` наследуется от :class:`~callbacks.Callback` и предназначен для записи и последующей визуализации различных параметров или метрик во время обучения.

**Основые атрибуты:**

- **d**: Словарь функций или параметров, которые будут записываться.

**Основые методы:**

- **`__init__(self, **d)`**: Инициализирует колбек с заданными параметрами для записи.
- **`before_fit(self, learn)`**: Инициализирует словарь записей для каждого параметра.
- **`after_batch(self, learn)`**: Добавляет текущие значения параметров в соответствующие списки.
- **`plot(self)`**: Отображает графики записанных параметров.

**Пример использования:**

.. code-block:: python

    from callbacks import RecorderCB

    recorder_cb = RecorderCB(loss=lambda cb: cb.learn.loss.item(), lr=lambda cb: cb.learn.opt.param_groups[0]['lr'])
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[recorder_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)

    # Визуализация записанных параметров
    recorder_cb.plot()

---

SingleBatchCB
-------------

Класс :class:`~callbacks.SingleBatchCB` наследуется от :class:`~callbacks.Callback` и предназначен для остановки обучения после обработки первого батча. Это может быть полезно для тестирования или отладки модели.

**Основые атрибуты:**

- **order**: Определяет порядок выполнения колбека относительно других колбеков. Устанавливается как `1`.

**Основые методы:**

- **`after_batch(self, learn)`**: Вызывает исключение `CancelFitException` после обработки первого батча, что приводит к остановке обучения.

**Пример использования:**

.. code-block:: python

    from callbacks import SingleBatchCB

    single_batch_cb = SingleBatchCB()
    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=[single_batch_cb],
        dataloaders=(train_dataloader, valid_dataloader)
    )
    learner.fit(epochs=10)  # Обучение остановится после первого батча

---

with_cbs
--------

Класс-декоратор :class:`~callbacks.with_cbs` предназначен для обёртывания методов класса `Learner` с целью автоматического вызова соответствующих методов колбеков перед и после выполнения метода.

**Основые атрибуты:**

- **nm**: Имя метода, к которому применяется декоратор.

**Основые методы:**

- **`__init__(self, nm)`**: Инициализирует декоратор с заданным именем метода.
- **`__call__(self, f)`**: Оборачивает функцию `f`, добавляя вызовы колбеков до и после её выполнения, а также обработку исключений.

**Пример использования:**

.. code-block:: python

    from callbacks import with_cbs

    @with_cbs('fit')
    def fit(self, *args, **kwargs):
        # Реализация метода fit
        pass

---