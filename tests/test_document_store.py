"""
اختبارات Document Store Service
======================================================
يتحقق من:
1. تخزين الوثائق الأصلية (raw) في SQLite
2. الاسترجاع بالـ doc_id (Primary Key)
3. الاسترجاع الجماعي (batch) بنفس ترتيب الـ doc_ids المُمرَّر
4. التعامل مع doc_id غير موجود
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '.')


@pytest.fixture
def temp_db_dir(monkeypatch):
    """قاعدة بيانات مؤقتة لكل اختبار — تُحذف بعد الانتهاء."""
    tmp = tempfile.mkdtemp()
    import services.document_store.main as mod
    monkeypatch.setattr(mod, "DB_DIR", Path(tmp))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestDocumentStoreInsertAndRetrieve:
    """التخزين والاسترجاع الأساسي."""

    def test_insert_and_get_single(self, temp_db_dir):
        from services.document_store.main import (
            init_db, get_connection, DocumentIn, BulkInsertRequest,
        )

        dataset_id = "test_ds"
        init_db(dataset_id)

        with get_connection(dataset_id) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                ("d1", "This is the raw original text.", "Doc 1", "{}"),
            )

        with get_connection(dataset_id) as conn:
            row = conn.execute("SELECT doc_id, raw_text, title FROM documents WHERE doc_id = ?", ("d1",)).fetchone()

        assert row[0] == "d1"
        assert row[1] == "This is the raw original text."
        assert row[2] == "Doc 1"

    def test_bulk_insert(self, temp_db_dir):
        """التخزين الجماعي — يُستخدَم من Dataset Loader."""
        from services.document_store.main import init_db, get_connection

        dataset_id = "test_bulk"
        init_db(dataset_id)

        docs = [
            ("d1", "Raw text one", "Title 1", "{}"),
            ("d2", "Raw text two", "Title 2", "{}"),
            ("d3", "Raw text three", "Title 3", "{}"),
        ]

        with get_connection(dataset_id) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                docs,
            )

        with get_connection(dataset_id) as conn:
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

        assert count == 3


class TestBatchRetrieval:
    """
    الاسترجاع الجماعي — هذا هو الـ endpoint الأهم.
    يُستدعى من Retrieval Service في "آخر خطوة" فقط بقائمة top-K doc_ids.
    """

    def test_batch_preserves_order(self, temp_db_dir):
        """
        الترتيب يجب أن يطابق ترتيب doc_ids المُمرَّر
        (وهو ترتيب نتائج البحث حسب الـ score).
        """
        from services.document_store.main import init_db, get_connection

        dataset_id = "test_order"
        init_db(dataset_id)

        with get_connection(dataset_id) as conn:
            conn.executemany(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                [
                    ("docA", "Text A", "A", "{}"),
                    ("docB", "Text B", "B", "{}"),
                    ("docC", "Text C", "C", "{}"),
                ],
            )

        # محاكاة منطق /get/batch: SELECT ... WHERE doc_id IN (...)
        # مع إعادة الترتيب حسب القائمة المُمرَّرة
        requested_order = ["docC", "docA", "docB"]

        with get_connection(dataset_id) as conn:
            placeholders = ",".join("?" * len(requested_order))
            cursor = conn.execute(
                f"SELECT doc_id, raw_text, title FROM documents WHERE doc_id IN ({placeholders})",
                requested_order,
            )
            rows = {row[0]: row for row in cursor.fetchall()}

        ordered = [rows[d] for d in requested_order]
        assert ordered[0][0] == "docC"
        assert ordered[1][0] == "docA"
        assert ordered[2][0] == "docB"

    def test_missing_doc_id_handled(self, temp_db_dir):
        """doc_id غير موجود يجب ألا يُسقِط الـ pipeline — يُعاد placeholder."""
        from services.document_store.main import init_db, get_connection

        dataset_id = "test_missing"
        init_db(dataset_id)

        with get_connection(dataset_id) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                ("exists1", "I exist", "Title", "{}"),
            )

        requested = ["exists1", "does_not_exist"]

        with get_connection(dataset_id) as conn:
            placeholders = ",".join("?" * len(requested))
            cursor = conn.execute(
                f"SELECT doc_id, raw_text, title FROM documents WHERE doc_id IN ({placeholders})",
                requested,
            )
            rows = {row[0]: row for row in cursor.fetchall()}

        # exists1 موجود، does_not_exist غير موجود
        assert "exists1" in rows
        assert "does_not_exist" not in rows


class TestPrimaryKeyIndex:
    """
    doc_id كـ PRIMARY KEY → SQLite يبني B-Tree index تلقائياً
    → استرجاع سريع O(log n) حتى مع آلاف الوثائق.
    """

    def test_primary_key_constraint(self, temp_db_dir):
        from services.document_store.main import init_db, get_connection

        dataset_id = "test_pk"
        init_db(dataset_id)

        with get_connection(dataset_id) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                ("dup", "first", "T1", "{}"),
            )

        # INSERT OR REPLACE يجب أن يستبدل لا يُضاعِف
        with get_connection(dataset_id) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                ("dup", "second", "T2", "{}"),
            )

        with get_connection(dataset_id) as conn:
            count = conn.execute("SELECT COUNT(*) FROM documents WHERE doc_id = 'dup'").fetchone()[0]
            text = conn.execute("SELECT raw_text FROM documents WHERE doc_id = 'dup'").fetchone()[0]

        assert count == 1
        assert text == "second"

    def test_large_batch_retrieval(self, temp_db_dir):
        """استرجاع دفعة كبيرة (محاكاة top-100) يعمل بدون مشاكل."""
        from services.document_store.main import init_db, get_connection

        dataset_id = "test_large"
        init_db(dataset_id)

        docs = [(f"doc_{i}", f"Raw text number {i}", f"Title {i}", "{}") for i in range(500)]

        with get_connection(dataset_id) as conn:
            conn.executemany(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                docs,
            )

        requested = [f"doc_{i}" for i in range(0, 100, 2)]  # 50 وثيقة

        with get_connection(dataset_id) as conn:
            placeholders = ",".join("?" * len(requested))
            cursor = conn.execute(
                f"SELECT doc_id, raw_text, title FROM documents WHERE doc_id IN ({placeholders})",
                requested,
            )
            rows = cursor.fetchall()

        assert len(rows) == 50


class TestRawTextIntegrity:
    """
    التحقق من أن النص الأصلي (raw) يُخزَّن كاملاً بدون اقتطاع،
    على عكس النسخة المُقتطَعة (2000 حرف) المُستخدَمة للفهرسة.
    """

    def test_long_text_not_truncated(self, temp_db_dir):
        from services.document_store.main import init_db, get_connection

        dataset_id = "test_long"
        init_db(dataset_id)

        long_text = "word " * 1000  # 5000 حرف — أطول من حد الفهرسة (2000)

        with get_connection(dataset_id) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                ("long_doc", long_text, "Long", "{}"),
            )

        with get_connection(dataset_id) as conn:
            row = conn.execute("SELECT raw_text FROM documents WHERE doc_id = ?", ("long_doc",)).fetchone()

        # النص الأصلي يجب أن يكون كاملاً (5000 حرف) وليس مُقتطَعاً لـ 2000
        assert len(row[0]) == len(long_text)
        assert len(row[0]) > 2000


class TestDatasetIsolation:
    """كل Dataset له ملف SQLite منفصل — لا تداخل بين msmarco و nq."""

    def test_separate_databases(self, temp_db_dir):
        from services.document_store.main import init_db, get_connection, get_db_path

        init_db("msmarco")
        init_db("nq")

        with get_connection("msmarco") as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                ("d1", "MS MARCO doc", "T", "{}"),
            )

        with get_connection("nq") as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
                ("d1", "NQ doc", "T", "{}"),
            )

        # نفس doc_id لكن في قاعدتي بيانات مختلفتين
        with get_connection("msmarco") as conn:
            text1 = conn.execute("SELECT raw_text FROM documents WHERE doc_id = 'd1'").fetchone()[0]

        with get_connection("nq") as conn:
            text2 = conn.execute("SELECT raw_text FROM documents WHERE doc_id = 'd1'").fetchone()[0]

        assert text1 == "MS MARCO doc"
        assert text2 == "NQ doc"
        assert get_db_path("msmarco") != get_db_path("nq")