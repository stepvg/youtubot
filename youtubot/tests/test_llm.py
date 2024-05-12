import unittest
from bot.llm import YouTuDoc

class TestYouTuDoc(unittest.TestCase):

    def setUp(self):
        # Создаем объекты для тестирования
        self.doc1 = YouTuDoc(page_content="Hello. world", metadata={'duration': 10, 'start': 0})
        self.doc2 = YouTuDoc(page_content="This is a test.", metadata={'duration': 20, 'start': 10})

    def test_split(self):
        # Проверяем метод split()
        doc1_first, doc1_second = self.doc1.split(3)
        self.assertEqual(doc1_first.page_content, "")
        self.assertEqual(doc1_second.page_content, "Hello. world")
        self.assertEqual(doc1_first.metadata['duration'], 0)
        self.assertEqual(doc1_second.metadata['duration'], 10)

        doc1_first, doc1_second = self.doc1.split(9)
        self.assertEqual(doc1_first.page_content, "Hello.")
        self.assertEqual(doc1_second.page_content, " world")
        self.assertEqual(doc1_first.metadata['duration'], 5)
        self.assertEqual(doc1_second.metadata['start'], 5)

        doc2_first, doc2_second = self.doc2.split(10)
        self.assertEqual(doc2_first.page_content, "")
        self.assertEqual(doc2_second.page_content, "This is a test.")
        self.assertEqual(doc2_first.metadata['start'], 10)
        self.assertEqual(doc2_second.metadata['duration'], 20)

    def test_zero_from(self):
        # Проверяем метод zero_from()
        zero_doc = YouTuDoc.zero_from(self.doc1)
        self.assertEqual(zero_doc.page_content, '')
        self.assertEqual(zero_doc.metadata['duration'], 0)
        self.assertEqual(zero_doc.metadata['start'], 0)

    def test_copy_from(self):
        # Проверяем метод copy_from()
        copied_doc = YouTuDoc.copy_from(self.doc2)
        self.assertEqual(copied_doc.page_content, "This is a test.")
        self.assertEqual(copied_doc.metadata['duration'], 20)
        self.assertEqual(copied_doc.metadata['start'], 10)

    def test_size(self):
        # Проверяем метод size()
        self.assertEqual(self.doc1.size(), 12)
        self.assertEqual(self.doc2.size(), 15)

    def test_add(self):
        # Проверяем метод __add__()
        combined_doc = self.doc1 + self.doc2
        self.assertEqual(combined_doc.page_content, "Hello. world This is a test.")
        self.assertEqual(combined_doc.metadata['duration'], 30)


if __name__ == '__main__':
    unittest.main()

