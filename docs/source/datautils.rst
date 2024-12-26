Data Utilities
=============

Утилиты для работы с данными, облегчающие обработку, загрузку и предобработку данных в библиотеке **Diffusion**.

.. toctree::
   :maxdepth: 2
   :caption: Содержание

   collate_dict
   dataloaders
   random_copy
   random_erase

Collate Dictionary Functions
----------------------------

.. function:: collate_dict(dataset: Dataset) -> Callable

    Объединяет батчи данных в формате словаря.

    :param dataset: Набор данных, содержащий определенные особенности (features), которые необходимо объединить.
    :type dataset: Dataset
    :return: Функция, принимающая список элементов батча и возвращающая объединенный словарь.
    :rtype: Callable

    **Пример использования:**

    .. code-block:: python

        from data_utils.collate_dict import collate_dict
        from datasets import Dataset
        from torch.utils.data import DataLoader

        # Инициализация набора данных
        dataset = Dataset(...)

        # Создание функции объединения батчей
        collate_fn = collate_dict(dataset)

        # Создание загрузчика данных с использованием функции объединения
        data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)


.. function:: collate_ddpm(b, alphabar=None, xl=None) -> Any

    Объединяет батч данных и добавляет шум для моделей DDPM.

    :param b: Батч данных.
    :type b: List
    :param alphabar: Параметр, определяющий уровень добавляемого шума. По умолчанию `None`.
    :type alphabar: float, опционально
    :param xl: Ключ для доступа к определенному элементу в батче данных. По умолчанию `None`.
    :type xl: str, опционально
    :return: Результат применения функции `noisify` к объединенному батчу данных.
    :rtype: Any

    **Пример использования:**

    .. code-block:: python

        from data_utils.collate_dict import collate_ddpm
        from torch.utils.data import DataLoader

        # Создание загрузчика данных с добавлением шума
        data_loader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=lambda b: collate_ddpm(b, alphabar=0.5, xl='data')
        )


.. function:: dl_ddpm(ds, bs=None, collate_fn=None, nw=None) -> DataLoader

    Создает загрузчик данных (`DataLoader`) для моделей DDPM с указанными параметрами.

    :param ds: Набор данных.
    :type ds: Dataset
    :param bs: Размер батча. По умолчанию `None`, что использует значение по умолчанию в `DataLoader`.
    :type bs: int, опционально
    :param collate_fn: Функция для объединения батча. По умолчанию `None`, что использует `default_collate`.
    :type collate_fn: Callable, опционально
    :param nw: Количество рабочих процессов для загрузки данных. По умолчанию `None`, что использует значение по умолчанию в `DataLoader`.
    :type nw: int, опционально
    :return: Объект загрузчика данных с заданными параметрами.
    :rtype: DataLoader

    **Пример использования:**

    .. code-block:: python

        from data_utils.collate_dict import dl_ddpm, collate_ddpm
        from torch.utils.data import DataLoader

        # Создание загрузчика данных с использованием dl_ddpm
        data_loader = dl_ddpm(
            ds=dataset,
            bs=32,
            collate_fn=lambda b: collate_ddpm(b, alphabar=0.5, xl='data'),
            nw=4
        )

DataLoaders Classes
-------------------

.. class:: DataLoaders(*dataloaders)

    Класс `DataLoaders` предназначен для хранения и управления несколькими загрузчиками данных, такими как тренировочный и валидационный.

    :param \*dataloaders: Загрузчики данных. Ожидается, что первыми двумя будут тренировочный и валидационный загрузчики.
    :type \*dataloaders: DataLoader

    **Пример использования:**

    .. code-block:: python

        from data_utils.dataloaders import DataLoaders
        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn)

        data_loaders = DataLoaders(train_loader, valid_loader)


.. class:: DataLoaders.from_dd(dataset_dict: dict, batch_size: int, num_workers: int, as_tuple: bool=True) -> 'DataLoaders'

    Создает экземпляр `DataLoaders` из словаря наборов данных, автоматически создавая соответствующие `DataLoader` объекты.

    :param dataset_dict: Словарь наборов данных, где ключи — названия (например, 'train', 'valid'), а значения — объекты `Dataset`.
    :type dataset_dict: dict
    :param batch_size: Размер батча для всех загрузчиков данных.
    :type batch_size: int
    :param num_workers: Количество рабочих процессов для загрузки данных.
    :type num_workers: int
    :param as_tuple: Если `True`, возвращает загрузчики данных как кортеж. По умолчанию `True`.
    :type as_tuple: bool, опционально
    :return: Экземпляр `DataLoaders` с созданными загрузчиками данных.
    :rtype: DataLoaders

    **Пример использования:**

    .. code-block:: python

        from data_utils.dataloaders import DataLoaders
        from data_utils.collate_dict import collate_dict
        from torch.utils.data import Dataset

        dataset_dict = {
            'train': train_dataset,
            'valid': valid_dataset
        }

        data_loaders = DataLoaders.from_dd(
            dataset_dict=dataset_dict,
            batch_size=32,
            num_workers=4
        )


.. class:: MultDL(dl: DataLoader, mult: int=2)

    Класс `MultDL` позволяет многократно итеративно проходить по одному и тому же загрузчику данных, умножая его длину.

    :param dl: Исходный загрузчик данных.
    :type dl: DataLoader
    :param mult: Множитель длины загрузчика данных. По умолчанию `2`.
    :type mult: int, опционально

    **Методы:**

    .. method:: __len__()

        Возвращает удвоенную длину исходного загрузчика данных.

        :return: Удвоенная длина загрузчика данных.
        :rtype: int

    .. method:: __iter__()

        Итератор, который многократно проходит по элементам исходного загрузчика данных.

        :yield: Элементы из исходного загрузчика данных.
        :rtype: Iterator

    **Пример использования:**

    .. code-block:: python

        from data_utils.dataloaders import MultDL
        from torch.utils.data import DataLoader

        original_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        mult_loader = MultDL(original_loader, mult=3)

        for batch in mult_loader:
            # Обработка батча
            pass


Random Copy Functions and Classes
---------------------------------

.. function:: random_copy(x, pct=0.2, max_num=4) -> Tensor

    Добавляет случайные копии области изображения для увеличения данных.

    :param x: Входные данные в виде тензора.
    :type x: Tensor
    :param pct: Процент размера области, которую нужно скопировать. По умолчанию `0.2`.
    :type pct: float, опционально
    :param max_num: Максимальное количество копий, которые будут добавлены. По умолчанию `4`.
    :type max_num: int, опционально
    :return: Тензор с добавленными случайными копиями областей.
    :rtype: Tensor

    **Пример использования:**

    .. code-block:: python

        from data_utils.random_copy import random_copy
        import torch

        # Пример тензора изображений (N, C, H, W)
        x = torch.randn(8, 3, 64, 64)

        # Добавление случайных копий
        x_augmented = random_copy(x, pct=0.2, max_num=4)


.. class:: RandCopy(pct=0.2, max_num=4)

    Модуль PyTorch для добавления случайных копий области изображения в процессе обучения.

    :param pct: Процент размера области, которую нужно скопировать. По умолчанию `0.2`.
    :type pct: float, опционально
    :param max_num: Максимальное количество копий, которые будут добавлены. По умолчанию `4`.
    :type max_num: int, опционально

    **Методы:**

    .. method:: forward(x: Tensor) -> Tensor

        Выполняет добавление случайных копий областей изображения.

        :param x: Входные данные в виде тензора.
        :type x: Tensor
        :return: Тензор с добавленными случайными копиями областей.
        :rtype: Tensor

    **Пример использования:**

    .. code-block:: python

        from data_utils.random_copy import RandCopy
        import torch

        # Инициализация модуля
        rand_copy = RandCopy(pct=0.2, max_num=4)

        # Пример тензора изображений (N, C, H, W)
        x = torch.randn(8, 3, 64, 64)

        # Применение модуля
        x_augmented = rand_copy(x)


Random Erase Functions and Classes
----------------------------------

.. function:: rand_erase(x, pct=0.2, max_num=4) -> Tensor

    Выполняет случайное стирание (замену) областей изображения для увеличения данных.

    :param x: Входные данные в виде тензора.
    :type x: Tensor
    :param pct: Процент размера области, которую нужно стереть. По умолчанию `0.2`.
    :type pct: float, опционально
    :param max_num: Максимальное количество стираний, которые будут выполнены. По умолчанию `4`.
    :type max_num: int, опционально
    :return: Тензор с выполненными случайными стираниями областей.
    :rtype: Tensor

    **Пример использования:**

    .. code-block:: python

        from data_utils.random_erase import rand_erase
        import torch

        # Пример тензора изображений (N, C, H, W)
        x = torch.randn(8, 3, 64, 64)

        # Выполнение случайного стирания
        x_augmented = rand_erase(x, pct=0.2, max_num=4)


.. class:: RandErase(pct=0.2, max_num=4)

    Модуль PyTorch для выполнения случайного стирания областей изображения в процессе обучения.

    :param pct: Процент размера области, которую нужно стереть. По умолчанию `0.2`.
    :type pct: float, опционально
    :param max_num: Максимальное количество стираний, которые будут выполнены. По умолчанию `4`.
    :type max_num: int, опционально

    **Методы:**

    .. method:: forward(x: Tensor) -> Tensor

        Выполняет случайное стирание областей изображения.

        :param x: Входные данные в виде тензора.
        :type x: Tensor
        :return: Тензор с выполненными случайными стираниями областей.
        :rtype: Tensor

    **Пример использования:**

    .. code-block:: python

        from data_utils.random_erase import RandErase
        import torch

        # Инициализация модуля
        rand_erase = RandErase(pct=0.2, max_num=4)

        # Пример тензора изображений (N, C, H, W)
        x = torch.randn(8, 3, 64, 64)

        # Применение модуля
        x_augmented = rand_erase(x)