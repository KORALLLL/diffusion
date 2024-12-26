Learners
========

Классы для управления процессом обучения моделей в библиотеке **Diffusion**. Эти классы обеспечивают гибкость и расширяемость тренировочного цикла, позволяя легко добавлять новые функции и изменять поведение обучения.

.. toctree::
   :maxdepth: 2
   :caption: Содержание

   learner
   train_learner
   momentum_learner



Learner
--------

:class: Learner(model, dls=(0,), loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.SGD)

    Базовый класс для управления процессом обучения модели.

    :param model: Модель, которую необходимо обучать.
    :type model: torch.nn.Module
    :param dls: Загрузчики данных для обучения и валидации. По умолчанию `(0,)`.
    :type dls: tuple, опционально
    :param loss_func: Функция потерь. По умолчанию `torch.nn.functional.mse_loss`.
    :type loss_func: callable, опционально
    :param lr: Начальная скорость обучения. По умолчанию `0.1`.
    :type lr: float, опционально
    :param cbs: Колбеки для расширения функциональности обучения. По умолчанию `None`.
    :type cbs: list или None, опционально
    :param opt_func: Функция оптимизатора. По умолчанию `torch.optim.SGD`.
    :type opt_func: callable, опционально

    **Описание:**

    Класс `Learner` управляет процессом обучения модели, включая обработку батчей, выполнение шагов оптимизации и управление колбеками. Он обеспечивает базовую структуру для реализации различных стратегий обучения.

    **Методы:**

    .. method:: fit(n_epochs=1, train=True, valid=True, cbs=None, lr=None)

        Запускает процесс обучения модели.

        :param n_epochs: Количество эпох обучения. По умолчанию `1`.
        :type n_epochs: int, опционально
        :param train: Флаг для включения обучения. По умолчанию `True`.
        :type train: bool, опционально
        :param valid: Флаг для включения валидации. По умолчанию `True`.
        :type valid: bool, опционально
        :param cbs: Дополнительные колбеки для этого вызова обучения. По умолчанию `None`.
        :type cbs: list или None, опционально
        :param lr: Скорость обучения, переопределяющая `self.lr`. По умолчанию `None`.
        :type lr: float или None, опционально

        **Пример использования:**

        .. code-block:: python

            from learners.learner import Learner
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            # Инициализация модели
            model = nn.Linear(10, 1)

            # Создание наборов данных
            train_ds = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
            valid_ds = TensorDataset(torch.randn(20, 10), torch.randn(20, 1))

            # Создание загрузчиков данных
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            valid_loader = DataLoader(valid_ds, batch_size=32)

            # Инициализация Learner
            learner = Learner(model=model, dls=(train_loader, valid_loader), lr=0.01)

            # Запуск обучения
            learner.fit(n_epochs=5, train=True, valid=True)

    .. method:: one_epoch(training)

        Выполняет одну эпоху обучения или валидации.

        :param training: Флаг, указывающий, выполняется ли обучение (`True`) или валидация (`False`).
        :type training: bool

        **Пример использования:**

        .. code-block:: python

            # Выполнение одной эпохи обучения
            learner.one_epoch(training=True)

            # Выполнение одной эпохи валидации
            learner.one_epoch(training=False)

    .. method:: callback(method_nm)

        Выполняет метод колбека с указанным именем.

        :param method_nm: Имя метода колбека, который необходимо выполнить.
        :type method_nm: str

    .. method:: __getattr__(name)

        Перенаправляет вызовы методов `predict`, `get_loss`, `backward`, `step`, `zero_grad` к соответствующим колбекам.

        :param name: Имя атрибута.
        :type name: str

    .. property:: training

        Возвращает состояние модели: обучается (`True`) или нет (`False`).

        :rtype: bool

    **Пример использования колбека:**

    .. code-block:: python

        from callbacks import Callback

        class PrintCallback(Callback):
            def before_fit(self, learn):
                print("Начало обучения")

            def after_fit(self, learn):
                print("Обучение завершено")

        # Инициализация Learner с колбеком
        learner = Learner(model, dls=(train_loader, valid_loader), cbs=[PrintCallback()])
        learner.fit(n_epochs=3)

---

TrainLearner
--------
:class: TrainLearner(model, dls, loss_func, lr=None, cbs=None, opt_func=optim.SGD)

    Класс `TrainLearner` наследуется от `Learner` и реализует основные методы для выполнения предсказаний, вычисления потерь и обратного распространения ошибки.

    :param model: Модель, которую необходимо обучать.
    :type model: torch.nn.Module
    :param dls: Загрузчики данных для обучения и валидации.
    :type dls: tuple или dict
    :param loss_func: Функция потерь.
    :type loss_func: callable
    :param lr: Скорость обучения. По умолчанию используется значение из `Learner`.
    :type lr: float, опционально
    :param cbs: Колбеки для расширения функциональности обучения. По умолчанию `None`.
    :type cbs: list или None, опционально
    :param opt_func: Функция оптимизатора. По умолчанию `torch.optim.SGD`.
    :type opt_func: callable, опционально

    **Описание:**

    `TrainLearner` реализует конкретные методы для выполнения предсказаний модели, вычисления функции потерь, обратного распространения ошибки и обновления параметров оптимизатора.

    **Методы:**

    .. method:: predict()

        Выполняет предсказание модели на текущем батче данных.

        **Пример использования:**

        .. code-block:: python

            learner.predict()

    .. method:: get_loss()

        Вычисляет функцию потерь на основе предсказаний модели и истинных значений.

        **Пример использования:**

        .. code-block:: python

            learner.get_loss()

    .. method:: backward()

        Выполняет обратное распространение ошибки для вычисленной функции потерь.

        **Пример использования:**

        .. code-block:: python

            learner.backward()

    .. method:: step()

        Обновляет параметры оптимизатора на основе вычисленных градиентов.

        **Пример использования:**

        .. code-block:: python

            learner.step()

    .. method:: zero_grad()

        Обнуляет градиенты оптимизатора перед новым шагом обучения.

        **Пример использования:**

        .. code-block:: python

            learner.zero_grad()

    **Пример использования `TrainLearner`:**

    .. code-block:: python

        from learners.train_learner import TrainLearner
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Инициализация модели
        model = nn.Linear(10, 1)

        # Создание наборов данных
        train_ds = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
        valid_ds = TensorDataset(torch.randn(20, 10), torch.randn(20, 1))

        # Создание загрузчиков данных
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=32)

        # Инициализация TrainLearner
        learner = TrainLearner(model=model, dls=(train_loader, valid_loader), loss_func=nn.MSELoss(), lr=0.01)

        # Запуск обучения
        learner.fit(n_epochs=5, train=True, valid=True)

---

MomentumLearner
--------

:class: MomentumLearner(model, dls, loss_func, lr=None, cbs=None, opt_func=optim.SGD, mom=0.85)

    Класс `MomentumLearner` наследуется от `TrainLearner` и добавляет поддержку момента при обновлении градиентов.

    :param model: Модель, которую необходимо обучать.
    :type model: torch.nn.Module
    :param dls: Загрузчики данных для обучения и валидации.
    :type dls: tuple или dict
    :param loss_func: Функция потерь.
    :type loss_func: callable
    :param lr: Скорость обучения. По умолчанию используется значение из `TrainLearner`.
    :type lr: float, опционально
    :param cbs: Колбеки для расширения функциональности обучения. По умолчанию `None`.
    :type cbs: list или None, опционально
    :param opt_func: Функция оптимизатора. По умолчанию `torch.optim.SGD`.
    :type opt_func: callable, опционально
    :param mom: Коэффициент момента. По умолчанию `0.85`.
    :type mom: float, опционально

    **Описание:**

    `MomentumLearner` расширяет функциональность `TrainLearner`, добавляя момент при обновлении градиентов оптимизатора. Это помогает ускорить сходимость обучения и стабилизировать процесс оптимизации.

    **Методы:**

    .. method:: zero_grad()

        Обнуляет градиенты оптимизатора, умножая их на коэффициент момента, вместо полного обнуления. Это сохраняет часть градиентов из предыдущих шагов, способствуя более стабильному обучению.

        **Пример использования:**

        .. code-block:: python

            learner.zero_grad()

    **Пример использования `MomentumLearner`:**

    .. code-block:: python

        from learners.momentum_learner import MomentumLearner
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Инициализация модели
        model = nn.Linear(10, 1)

        # Создание наборов данных
        train_ds = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
        valid_ds = TensorDataset(torch.randn(20, 10), torch.randn(20, 1))

        # Создание загрузчиков данных
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=32)

        # Инициализация MomentumLearner с моментом 0.9
        learner = MomentumLearner(
            model=model,
            dls=(train_loader, valid_loader),
            loss_func=nn.MSELoss(),
            lr=0.01,
            mom=0.9
        )

        # Запуск обучения
        learner.fit(n_epochs=5, train=True, valid=True)