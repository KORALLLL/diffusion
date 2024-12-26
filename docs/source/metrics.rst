Metrics
=======

Модули для работы с метриками в библиотеке **Diffusion**. Содержит реализации стандартных метрик для оценки моделей, а также инструменты для расчёта FID и KID для изображений.

.. toctree::
   :maxdepth: 2
   :caption: Содержание

   Metric
   Accuracy
   ImageEval

Metric
------

 :class: Metric

    Базовый класс для создания метрик.

    **Описание:**

    `Metric` предоставляет интерфейс для создания пользовательских метрик. Вы можете наследоваться от этого класса и переопределять метод `calc` для реализации своей логики расчёта метрик.

    **Методы:**

    .. method:: reset()

        Сбрасывает внутренние значения метрики. Полезно для начала расчётов заново.

        **Пример использования:**

        .. code-block:: python

            metric = Metric()
            metric.reset()

    .. method:: add(inp, targ=None, n=1)

        Добавляет значение метрики на основе входных данных.

        :param inp: Предсказания модели.
        :type inp: tensor
        :param targ: Истинные значения (по умолчанию `None`).
        :type targ: tensor, опционально
        :param n: Вес метрики (например, размер батча).
        :type n: int

        **Пример использования:**

        .. code-block:: python

            metric.add(predictions, targets, n=batch_size)

    .. property:: value

        Возвращает текущее значение метрики.

        **Пример использования:**

        .. code-block:: python

            current_value = metric.value

    .. method:: calc(inps, targs)

        Метод расчёта значения метрики. Переопределите этот метод в подклассах для реализации пользовательской логики.

Accuracy
--------

 :class: Accuracy(Metric)

    Класс для вычисления точности (accuracy).

    **Описание:**

    `Accuracy` наследуется от `Metric` и реализует метод `calc`, который вычисляет точность как среднее значение совпадений предсказаний и целевых значений.

    **Методы:**

    .. method:: calc(inps, targs)

        Вычисляет точность (accuracy).

        :param inps: Предсказания модели.
        :type inps: tensor
        :param targs: Истинные значения.
        :type targs: tensor
        :return: Точность.
        :rtype: tensor

        **Пример использования:**

        .. code-block:: python

            from metrics.accuracy import Accuracy

            accuracy = Accuracy()
            acc_value = accuracy.calc(predictions, targets)

ImageEval
---------

 :class: ImageEval

    Класс для расчёта метрик FID и KID для изображений.

    :param model: Обучаемая модель.
    :type model: torch.nn.Module
    :param dls: Загрузчики данных.
    :type dls: DataLoaders
    :param cbs: Колбеки для расширения функциональности. По умолчанию `None`.
    :type cbs: list или None

    **Описание:**

    `ImageEval` предоставляет методы для расчёта FID (Frechet Inception Distance) и KID (Kernel Inception Distance), используемых для оценки качества генерируемых изображений.

    **Методы:**

    .. method:: fid(samp)

        Вычисляет FID для заданного набора образцов.

        :param samp: Образцы, для которых нужно рассчитать FID.
        :type samp: tensor
        :return: Значение FID.
        :rtype: float

        **Пример использования:**

        .. code-block:: python

            from metrics.fid_kid import ImageEval

            image_eval = ImageEval(model, dls)
            fid_value = image_eval.fid(samples)

    .. method:: kid(samp)

        Вычисляет KID для заданного набора образцов.

        :param samp: Образцы, для которых нужно рассчитать KID.
        :type samp: tensor
        :return: Значение KID.
        :rtype: float

        **Пример использования:**

        .. code-block:: python

            kid_value = image_eval.kid(samples)