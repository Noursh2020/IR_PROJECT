"""
conftest.py — يُحدد مسار المشروع لـ pytest
ضعه في مجلد ir_project (نفس مستوى مجلد services)
"""
import sys
import os

# أضف مجلد المشروع الرئيسي لـ Python path
# حتى يتعرف pytest على مجلد services
sys.path.insert(0, os.path.dirname(__file__))