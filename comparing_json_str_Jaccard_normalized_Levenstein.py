import json
import os
from typing import List, Tuple, Any


class BaseEval:
    def __init__(self, documents_path: str):
        """A class to evaluate differences between text and OCR text from documents."""
        self.documents = self.load_documents(documents_path)
        if not self.documents:
            raise ValueError("The documents list is empty.")

    def load_documents(self, path: str) -> List[Tuple[str, str]]:
        """
        Loads documents from a JSON file.

        Args:
            path (str): The file path to the JSON file.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing (text, ocr_text).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file does not contain a valid list of documents
                        or is not valid JSON.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

        if not isinstance(data, list):
            raise ValueError("JSON root must be a list of documents")

        docs: List[Tuple[str, str]] = []
        for idx, doc in enumerate(data):
            try:
                docs.append(self.validate_document(doc))
            except ValueError as e:
                raise ValueError(f"Document at index {idx} is invalid: {e}")
        return docs

    def validate_document(self, doc: Any) -> Tuple[str, str]:
        """
        Validates that a document contains the required fields.

        Args:
            doc (dict): The document to validate.

        Returns:
            Tuple[str, str]: (text, ocr_text)

        Raises:
            ValueError: If the document or fields are invalid.
        """
        if not isinstance(doc, dict):
            raise ValueError("Document must be a dictionary")
        if 'text' not in doc:
            raise ValueError("Missing 'text' field")
        if 'ocr_text' not in doc:
            raise ValueError("Missing 'ocr_text' field'")

        text = doc['text']
        ocr_text = doc['ocr_text']
        if not isinstance(text, str):
            raise ValueError("'text' field must be a string")
        if not isinstance(ocr_text, str):
            raise ValueError("'ocr_text' field must be a string")

        return text, ocr_text

    def _eval_func(self, text: str, ocr_text: str) -> float:
        """
        Evaluates the difference between text and OCR text by per-character comparison.

        Args:
            text (str): The original text.
            ocr_text (str): The OCR extracted text.

        Returns:
            float: The number of differing characters.
        """
        
        diffs = sum(c1 != c2 for c1, c2 in zip(text, ocr_text))
        diffs += abs(len(text) - len(ocr_text))
        return float(diffs)
        
    def evaluate(self, limit: int = None) -> List[Tuple[float, str, str]]:
        """
        Evaluates all documents and returns the differences, sorted descending by diff count.

        Args:
            limit (int, optional): Max number of results to return. If None, returns all.

        Returns:
            List[Tuple[float, str, str]]: List of (difference, text, ocr_text),
                                          sorted by difference descending.
        """
        results: List[Tuple[float, str, str]] = []
        for text, ocr in self.documents:
            diff = self._eval_func(text, ocr)
            results.append((diff, text, ocr))

        # stable sort by diff descending; Python's sort is stable
        results.sort(key=lambda tpl: tpl[0], reverse=True)

        if limit is not None:
            return results[:limit]
        return results


class JaccardEval(BaseEval):
    """
    Оценивает разницу между текстом и OCR-текстом как 1 - Jaccard similarity по словам.
    """
    def _eval_func(self, text: str, ocr_text: str) -> float:
        """
        1 - коэффициент Жаккара между множествами слов оригинала и OCR-текста.
        """
        # разбиваем по пробелам
        words1 = set(text.split())
        words2 = set(ocr_text.split())

        if not words1 and not words2:
            # оба пусты — считаем полное совпадение => метрика 0
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        jaccard_sim = len(intersection) / len(union)
        return 1.0 - jaccard_sim


class LevenshteinEval(BaseEval):
    """
    Оценивает разницу между текстом и OCR-текстом как расстояние Левенштейна
    (алгоритм Вагнера–Фишера).
    """
    def _eval_func(self, text: str, ocr_text: str) -> float:
        """
        Считает расстояние Левенштейна между text и ocr_text.
        """
        len_t, len_o = len(text), len(ocr_text)
        # если один из текстов пуст, возвращаем длину другого
        if len_t == 0:
            return float(len_o)
        if len_o == 0:
            return float(len_t)

        # инициализируем DP-массивы
        prev_row = list(range(len_o + 1))
        curr_row = [0] * (len_o + 1)

        for i in range(1, len_t + 1):
            curr_row[0] = i
            c1 = text[i - 1]
            for j in range(1, len_o + 1):
                c2 = ocr_text[j - 1]
                cost = 0 if c1 == c2 else 1
                curr_row[j] = min(
                    prev_row[j] + 1,        # удаление
                    curr_row[j - 1] + 1,     # вставка
                    prev_row[j - 1] + cost   # замена
                )
            prev_row, curr_row = curr_row, prev_row

        return float(prev_row[len_o])

class NormLevEval(BaseEval):
    """
    Оценивает нормализованное расстояние Левенштейна между текстом и OCR-текстом.
    Нормализация производится делением на максимально возможную стоимость,
    равную длине большей строки, умноженной на максимальную из трех стоимостей.
    """
    def __init__(
        self,
        documents_path: str,
        insert_cost: float = 1,
        delete_cost: float = 1,
        substitute_cost: float = 1,
    ):
        """
        Инициализирует NormLevEval с путём к документам и параметрами стоимостей.

        Raises:
            ValueError: Если любой параметр стоимости не в диапазоне [0, 2].
        """
        # Проверяем стоимости
        for name, cost in (
            ('insert_cost', insert_cost),
            ('delete_cost', delete_cost),
            ('substitute_cost', substitute_cost)
        ):
            if not (0 <= cost <= 2):
                raise ValueError(f"{name} must be in [0, 2], got {cost}")

        super().__init__(documents_path)
        self.insert_cost = insert_cost
        self.delete_cost = delete_cost
        self.substitute_cost = substitute_cost

    def _eval_func(self, text: str, ocr_text: str) -> float:
        """
        Считает нормализованное расстояние Левенштейна между text и ocr_text.
        """
        len_t, len_o = len(text), len(ocr_text)
        # если один из текстов пуст
        if len_t == 0:
            dist = len_o * self.insert_cost
        elif len_o == 0:
            dist = len_t * self.delete_cost
        else:
            # инициализируем DP-строки с учётом стоимостей вставки
            prev_row = [j * self.insert_cost for j in range(len_o + 1)]
            curr_row = [0.0] * (len_o + 1)

            for i in range(1, len_t + 1):
                curr_row[0] = i * self.delete_cost
                c1 = text[i - 1]
                for j in range(1, len_o + 1):
                    c2 = ocr_text[j - 1]
                    cost = 0.0 if c1 == c2 else self.substitute_cost
                    curr_row[j] = min(
                        prev_row[j] + self.delete_cost,        # удаление
                        curr_row[j - 1] + self.insert_cost,     # вставка
                        prev_row[j - 1] + cost                   # замена
                    )
                prev_row, curr_row = curr_row, prev_row
            dist = prev_row[len_o]

        # нормируем
        max_cost = max(self.insert_cost, self.delete_cost, self.substitute_cost)
        max_len = max(len_t, len_o)
        denom = max_len * max_cost
        return 0.0 if denom == 0 else dist / denom
