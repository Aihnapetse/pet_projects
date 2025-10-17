import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array,
              over_penalty: float = 2.0,
              under_penalty: float = 1.0) -> float:
    """
    Асимметричная функция потерь по абсолютной ошибке.
    Переоценка штрафуется сильнее.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    diff = y_pred - y_true

    error = np.where(
        diff > 0,  # переоценка
        over_penalty * np.abs(diff),
        under_penalty * np.abs(diff)
    )
    return float(np.mean(error))
