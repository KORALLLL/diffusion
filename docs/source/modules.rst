Modules
=======

Модули, предоставляющие основные строительные блоки и вспомогательные функции для работы с нейронными сетями в библиотеке **Diffusion**.

.. toctree::
   :maxdepth: 2
   :caption: Содержание

   conv
   general_relu
   get_model
   init_ddpm
   init_weights
   lsuv_init
   resnet
   summary

Conv
----

:function: conv(input_channels, output_channels, ks=3, stride=2, act=nn.ReLU, norm=None, bias=False)

    Создает свёрточный блок с возможным добавлением активации и нормализации.

    :param input_channels: Количество входных каналов.
    :type input_channels: int
    :param output_channels: Количество выходных каналов.
    :type output_channels: int
    :param ks: Размер ядра свёртки. По умолчанию `3`.
    :type ks: int, опционально
    :param stride: Шаг свёртки. По умолчанию `2`.
    :type stride: int, опционально
    :param act: Активационная функция. По умолчанию `nn.ReLU`.
    :type act: callable, опционально
    :param norm: Функция нормализации. По умолчанию `None`.
    :type norm: callable, опционально
    :param bias: Использовать ли смещение (bias). По умолчанию `False`.
    :type bias: bool, опционально
    :return: Последовательность (`nn.Sequential`) слоев свёртки, нормализации и активации.
    :rtype: nn.Sequential

    **Пример использования:**

    .. code-block:: python

        from modules.conv import conv
        import torch.nn as nn

        conv_block = conv(3, 16, ks=3, stride=1, act=nn.ReLU, norm=nn.BatchNorm2d)

GeneralRelu
-----------

:class: GeneralRelu(leak=None, sub=None, maxv=None)

    Расширяемая версия функции активации ReLU с дополнительными параметрами.

    :param leak: Коэффициент утечки для Leaky ReLU. Если `None`, используется стандартный ReLU.
    :type leak: float, опционально
    :param sub: Сдвиг (вычитание) значений после активации. По умолчанию `None`.
    :type sub: float, опционально
    :param maxv: Максимальное значение для обрезки (clamping). По умолчанию `None`.
    :type maxv: float, опционально

    **Пример использования:**

    .. code-block:: python

        from modules.general_relu import GeneralRelu

        act = GeneralRelu(leak=0.1, sub=0.4, maxv=6.0)

GetModel
--------

:function: get_model(act=nn.ReLU, nfs=None, norm=None)

    Создает модель на основе последовательности свёрточных блоков.

    :param act: Активационная функция. По умолчанию `nn.ReLU`.
    :type act: callable, опционально
    :param nfs: Список количества фильтров для каждого свёрточного блока. По умолчанию `[1, 8, 16, 32, 64]`.
    :type nfs: list, опционально
    :param norm: Функция нормализации. По умолчанию `None`.
    :type norm: callable, опционально
    :return: Модель `nn.Sequential`.
    :rtype: nn.Sequential

    **Пример использования:**

    .. code-block:: python

        from modules.get_model import get_model

        model = get_model()

InitDDPM
--------

:function: init_ddpm(model)

    Инициализирует веса модели DDPM.

    :param model: Модель DDPM.
    :type model: torch.nn.Module

    **Пример использования:**

    .. code-block:: python

        from modules.init_ddpm import init_ddpm

        init_ddpm(ddpm_model)

InitWeights
-----------

:function: init_weights(m, leaky=0.)

    Инициализирует веса модели с использованием `kaiming_normal_`.

    :param m: Модуль для инициализации весов.
    :type m: torch.nn.Module
    :param leaky: Коэффициент утечки для Leaky ReLU. По умолчанию `0`.
    :type leaky: float, опционально

    **Пример использования:**

    .. code-block:: python

        from modules.init_weights import init_weights
        import torch.nn as nn

        model = nn.Conv2d(3, 16, kernel_size=3)
        model.apply(init_weights)

LSUVInit
--------

:function: lsuv_init(m, m_in, xb, model)

    Выполняет LSUV (Layer-Sequential Unit-Variance) инициализацию весов модели.

    :param m: Модуль, который нужно инициализировать.
    :type m: torch.nn.Module
    :param m_in: Входной модуль.
    :type m_in: torch.nn.Module
    :param xb: Входные данные.
    :type xb: torch.Tensor
    :param model: Модель для обучения.
    :type model: torch.nn.Module

    **Пример использования:**

    .. code-block:: python

        from modules.lsuv_init import lsuv_init

        lsuv_init(layer, layer_input, input_batch, model)

ResNet
------

:class: ResBlock(ni, nf, stride=1, ks=3, act=act_gr, norm=None)

    Реализация Residual блока.

    :param ni: Количество входных каналов.
    :type ni: int
    :param nf: Количество выходных каналов.
    :type nf: int
    :param stride: Шаг свёртки. По умолчанию `1`.
    :type stride: int, опционально
    :param ks: Размер ядра свёртки. По умолчанию `3`.
    :type ks: int, опционально
    :param act: Функция активации. По умолчанию `act_gr`.
    :type act: callable, опционально
    :param norm: Функция нормализации. По умолчанию `None`.
    :type norm: callable, опционально

    **Пример использования:**

    .. code-block:: python

        from modules.resnet import ResBlock

        res_block = ResBlock(64, 128, stride=2)

Summary
-------

:function: summary(self)

    Выводит краткое описание модели, включая информацию о модулях, размерах входов и выходов, а также количестве параметров.

    :param self: Экземпляр `Learner`.
    :type self: Learner

    **Пример использования:**

    .. code-block:: python

        from modules.summary import summary

        learner.summary()