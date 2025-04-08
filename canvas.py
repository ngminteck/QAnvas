import os
import re
import difflib
import time
import html
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import shutil
import torch
from canvasapi import Canvas

# Third-party libraries for embeddings and loaders
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

# For OCR extraction
from pdf2image import convert_from_path
import pytesseract
import requests

# Import Camelot for table extraction
import camelot


def list_all_models():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get("https://api.openai.com/v1/models", headers=headers)
    if response.status_code == 200:
        models = response.json()
        print("Available Models:")
        for model in models.get("data", []):
            print(model["id"])
    else:
        print("Error listing models:", response.text)


class CanvasManager:
    """
    A manager class to handle Canvas operations such as retrieving lecture slides,
    downloading files, managing timetables, assignments, announcements, etc.
    """
    multi_agent = None
    # Adjustable threshold for OCR fallback.
    OCR_WORD_THRESHOLD = 5

    def __init__(self, api_url: str, api_key: str):
        self.canvas = Canvas(api_url, api_key)
        self.api_url = api_url

    # ------------------------------------------------------
    # HELPER / UTILITY FUNCTIONS
    # ------------------------------------------------------
    @staticmethod
    def get_match_score(query: str, title: str) -> float:
        query_norm = re.sub(r'\W+', ' ', query).strip().lower()
        title_norm = re.sub(r'\W+', ' ', title).strip().lower()
        if query_norm in title_norm:
            return 1.0
        return difflib.SequenceMatcher(None, query_norm, title_norm).ratio()

    @staticmethod
    def strip_html(text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        return html.unescape(text)

    @staticmethod
    def _parse_date(date_str: str):
        parsed_date = None
        try:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                parsed_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            except Exception:
                parsed_date = None
        return parsed_date

    @staticmethod
    def sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "_", name)

    @classmethod
    def build_folder_path_map(cls, course) -> dict:
        folder_map = {}
        folders = list(course.get_folders())
        folder_dict = {folder.id: folder for folder in folders}

        def get_full_path(folder):
            if folder.parent_folder_id and folder.parent_folder_id in folder_dict:
                parent_folder = folder_dict[folder.parent_folder_id]
                return os.path.join(get_full_path(parent_folder), cls.sanitize_filename(folder.name))
            else:
                return cls.sanitize_filename(folder.name)

        for folder in folders:
            folder_map[folder.id] = get_full_path(folder)
        return folder_map

    @staticmethod
    def ocr_extract_text(pdf_path: str):
        images = convert_from_path(pdf_path)
        texts = []
        for image in images:
            text = pytesseract.image_to_string(image)
            texts.append(text)
        return texts

    @staticmethod
    def clean_text(text: str) -> str:
        text = CanvasManager.strip_html(text)
        text = CanvasManager.remove_sensitive_info(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def remove_sensitive_info(text: str) -> str:
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        text = re.sub(r'(\+?\d[\d\-\s().]{7,}\d)', '[REDACTED_PHONE]', text)
        return text

    # ======================================================
    # DOWNLOAD FILES FUNCTIONS
    # ======================================================
    def download_all_files_parallel(self, base_dir: str = "files", max_workers: int = None):
        if max_workers is None:
            max_workers = os.cpu_count() or 4
        courses = list(self.canvas.get_courses())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for course in courses:
                if getattr(course, "access_restricted_by_date", False):
                    continue
                course_name = self.sanitize_filename(course.name)
                try:
                    folder_path_map = self.build_folder_path_map(course)
                except Exception:
                    folder_path_map = {}
                try:
                    files = list(course.get_files())
                except Exception:
                    continue
                for f in files:
                    futures.append(executor.submit(self.download_file, f, course_name, folder_path_map, base_dir))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

    def download_file(self, file_obj, course_name, folder_path_map, base_dir):
        try:
            folder_id = getattr(file_obj, "folder_id", None)
            folder_name = folder_path_map.get(folder_id, "")
            target_dir = os.path.join(base_dir, course_name, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, str(file_obj.display_name))
            file_obj.download(file_path)
        except Exception:
            pass

    # ------------------------------------------------------
    # EMBEDDING INDEX BUILDING (with OCR fallback)
    # ------------------------------------------------------
    def build_embedding_index(self, index_dir: str = "chroma_index", base_dir: str = "files"):
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        cpu_cores = os.cpu_count() or 4
        start_time = time.time()
        selected_paths = []
        if not os.path.exists(base_dir):
            return None
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.pptx', '.doc', '.docx')):
                    selected_paths.append(os.path.join(root, file))
        if not selected_paths:
            return None
        documents = []
        max_workers = os.cpu_count() or 4

        def load_file(fp):
            try:
                docs = []
                if fp.lower().endswith('.pdf'):
                    loader = PyPDFLoader(fp)
                    docs = loader.load()
                    for idx, doc in enumerate(docs):
                        if len(doc.page_content.split()) < CanvasManager.OCR_WORD_THRESHOLD:
                            ocr_texts = CanvasManager.ocr_extract_text(fp)
                            if idx < len(ocr_texts):
                                doc.page_content = ocr_texts[idx]
                else:
                    loader = UnstructuredLoader(fp)
                    docs = loader.load()
                for doc in docs:
                    if doc.page_content:
                        doc.page_content = CanvasManager.clean_text(doc.page_content)
                    doc.metadata["file_path"] = fp
                    doc.metadata["canvas_link"] = f"{self.api_url}/files/{os.path.basename(fp)}"
                return docs
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_file, fp): fp for fp in selected_paths}
            for future in as_completed(futures):
                docs_from_file = future.result()
                documents.extend(docs_from_file)

        if not documents:
            return None

        pdf_docs = []
        non_pdf_docs = []
        for doc in documents:
            if doc.metadata.get("file_path", "").lower().endswith(".pdf"):
                pdf_docs.append(doc)
            else:
                non_pdf_docs.append(doc)
        if non_pdf_docs:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            non_pdf_docs = text_splitter.split_documents(non_pdf_docs)
        docs = pdf_docs + non_pdf_docs

        new_docs = []
        for doc in docs:
            if isinstance(doc, tuple) or not hasattr(doc, "metadata"):
                try:
                    doc = Document(page_content=doc[0], metadata=doc[1])
                except Exception:
                    continue
            new_docs.extend(filter_complex_metadata([doc]))
        docs = new_docs

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        try:
            vector_store = Chroma.from_documents(docs, embeddings, persist_directory=index_dir)
            vector_store.persist()
            elapsed_time = time.time() - start_time
            return vector_store
        except Exception:
            return None

    # ------------------------------------------------------
    # BUILD ALL SUMMARIES (using ChatGPT with OCR fallback if needed)
    # ------------------------------------------------------
    def build_all_summaries(self, base_dir: str = "files", summary_base_dir: str = "summary"):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.pptx', '.doc', '.docx')):
                    file_path = os.path.join(root, file)
                    try:
                        CanvasManager.create_summary_csv(file_path, summary_base_dir, base_dir)
                    except Exception:
                        pass

    # ------------------------------------------------------
    # Create Summary CSV File for a Given Document (Page by Page)
    # ------------------------------------------------------
    @classmethod
    def create_summary_csv(cls, file_path: str, summary_base_dir: str, base_dir: str):
        min_word_count = cls.OCR_WORD_THRESHOLD

        rel_path = os.path.relpath(file_path, base_dir)
        summary_file_path = os.path.join(summary_base_dir, rel_path)
        summary_file_path = os.path.splitext(summary_file_path)[0] + ".csv"

        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredLoader(file_path)
        try:
            docs = loader.load()
        except Exception:
            return

        if not docs:
            return

        os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)

        summaries = []
        need_update = False
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_rows = list(reader)
            except Exception:
                existing_rows = []
            if len(existing_rows) != len(docs):
                need_update = True
            else:
                for row in existing_rows:
                    summary_text = row.get("Summary", "").strip()
                    if len(summary_text.split()) < min_word_count:
                        need_update = True
                        break
                if not need_update:
                    return
                else:
                    for row in existing_rows:
                        summaries.append(row.get("Summary", "").strip())

        ocr_texts = None
        if file_path.lower().endswith('.pdf'):
            insufficient = any(len(cls.clean_text(doc.page_content).split()) < min_word_count for doc in docs)
            if insufficient:
                ocr_texts = cls.ocr_extract_text(file_path)

        for i, doc in enumerate(docs, start=1):
            text = cls.clean_text(doc.page_content)
            if file_path.lower().endswith('.pdf') and len(text.split()) < min_word_count and ocr_texts:
                if i - 1 < len(ocr_texts):
                    text = cls.clean_text(ocr_texts[i - 1])
            if not text.strip():
                new_summary = "[No content to summarize]"
            else:
                regenerate = False
                if i > len(summaries):
                    regenerate = True
                else:
                    existing = summaries[i - 1] if i - 1 < len(summaries) else ""
                    if len(existing.split()) < min_word_count:
                        regenerate = True
                if regenerate:
                    max_attempts = 3
                    attempt = 1
                    new_summary = ""
                    while attempt <= max_attempts:
                        new_summary = cls.summarize_page_chatgpt(
                            text,
                            page_number=i,
                            file_name=os.path.basename(file_path)
                        )
                        if len(new_summary.split()) >= min_word_count:
                            break
                        attempt += 1
                else:
                    new_summary = summaries[i - 1]
            if i - 1 < len(summaries):
                summaries[i - 1] = new_summary
            else:
                summaries.append(new_summary)

        try:
            with open(summary_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["Page Number", "Summary"])
                for i, summary in enumerate(summaries, start=1):
                    csvwriter.writerow([i, summary])
        except Exception:
            pass

    # ------------------------------------------------------
    # Summarize a Page Using OpenAI ChatCompletion API
    # ------------------------------------------------------
    @staticmethod
    def summarize_page_chatgpt(text: str, page_number: int, file_name: str) -> str:
        prompt = (
            f"Please summarize the following content from page {page_number} of the document '{file_name}' "
            f"in a concise paragraph. Focus on the key points:\n\n{text}"
        )
        try:
            llm = OpenAI(temperature=0.5, max_tokens=150)
            response = llm.invoke(prompt).strip()
            return response
        except Exception:
            return "[API Error]"

    # ======================================================
    # RETRIEVE LECTURE SLIDES FUNCTIONS
    # ======================================================
    def retrieve_lecture_slides_by_topic(self, topic: str, index_dir: str = "chroma_index", k: int = 100,
                                           filter_terms: list = None):
        def resolve_filter(filter_list, alias_mapping, threshold=0.8):
            resolved = []
            for item in filter_list:
                item_lower = item.lower()
                for canonical, aliases in alias_mapping.items():
                    if CanvasManager.get_match_score(item_lower, canonical.lower()) >= threshold:
                        resolved.append(canonical)
                        break
                    elif any(CanvasManager.get_match_score(item_lower, alias.lower()) >= threshold for alias in aliases):
                        resolved.append(canonical)
                        break
            return resolved

        course_aliases = {
            "ISY5004": ["Intelligent Sensing Systems", "Computer Vision", "CV", "ISS"],
            "EBA5004": ["Practical Language Processing", "NLP", "Nature Language Processing", "PLP"]
        }
        sub_module_aliases = {
            "VSD": ["Vision Systems", "VS"],
            "SRSD": ["Spatial Reasoning from Sensor Data"],
            "RTAVS": ["Real time Audio-Visual Sensing and Sense Making"],
            "TA": ["Text Analytics"],
            "NMSM": ["New Media and Sentiment Mining"],
            "TPML": ["Text Processing Using Machine Learning"],
            "CUI": ["Conversational Uls", "CNI"]
        }

        resolved_course = resolve_filter(filter_terms, course_aliases, threshold=0.8) if filter_terms else []
        resolved_submodule = resolve_filter(filter_terms, sub_module_aliases, threshold=0.8) if filter_terms else []

        file_paths = {
            "ISY5004": {
                "VSD": {"path": "files\\course files\\Vision Systems (6-10Jan 2025)"},
                "SRSD": {"path": "files\\course files\\Spatial Reasoning from Sensor Data (13-15Jan 2025)"},
                "RTAVS": {"path": "files\\course files\\Real-time Audio-Visual Sensing and Sense Making (20-23Jan 2025)"}
            },
            "EBA5004": {
                "TA": {"path": "files\\course files\\[PLP] Text Analytics (2025-02-10)"},
                "NMSM": {"path": "files\\course files\\EBA5004 Practical Language Processing [2420]\\01_NMSM"},
                "TPML": {"path": "files\\course files\\EBA5004 Practical Language Processing [2420]\\02_TPML"},
                "CUI": {"path": "files\\course files\\EBA5004 Practical Language Processing [2420]\\03_CNI"}
            }
        }

        courses_to_process = {}
        for course, submodules in file_paths.items():
            matching_submodules = {sm: details["path"] for sm, details in submodules.items() if sm in resolved_submodule}
            if matching_submodules:
                courses_to_process[course] = matching_submodules
            elif course in resolved_course:
                courses_to_process[course] = {sm: details["path"] for sm, details in submodules.items()}

        if not filter_terms:
            courses_to_process = {course: {sm: details["path"] for sm, details in submodules.items()} for course, submodules in file_paths.items()}

        if not courses_to_process:
            return []

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        if os.path.exists(index_dir):
            try:
                vector_store = Chroma(persist_directory=index_dir, embedding_function=embeddings)
            except Exception:
                return []
        else:
            vector_store = self.build_embedding_index(index_dir=index_dir)
            if vector_store is None:
                return []

        try:
            results = vector_store.similarity_search(topic, k=k)
        except Exception:
            return []
        if filter_terms:
            allowed_paths = []
            for course, submodules in courses_to_process.items():
                for submodule, allowed_path in submodules.items():
                    allowed_paths.append(allowed_path.lower())
            filtered_results = []
            for res in results:
                file_path = res.metadata.get("file_path", "").lower()
                if any(allowed_path in file_path for allowed_path in allowed_paths):
                    filtered_results.append(res)
            results = filtered_results

        top_results = results[:k]
        final_results = []
        for idx, res in enumerate(top_results, start=1):
            file_info = res.metadata.get("file_path", "Unknown file")
            page_info = res.metadata.get("page", "N/A")
            canvas_link = res.metadata.get("canvas_link", "Not available")
            filename = os.path.basename(file_info) if file_info != "Unknown file" else file_info
            final_results.append({
                "result_rank": idx,
                "content": res.page_content,
                "file_path": file_info,
                "filename": filename,
                "page": page_info,
                "canvas_link": canvas_link,
                "metadata": res.metadata
            })
        return final_results

    # ======================================================
    # TIMETABLE AND EXAM DATE FUNCTIONS
    # ======================================================
    def get_timetable_or_exam_date(self, full_time: bool, intake_year: int, course: str, query: str):
        if full_time:
            mode_folder = "MTech Full Time"
            intake_folder = f"Aug {intake_year} FT Intake"
        else:
            mode_folder = "MTech Part Time"
            intake_folder = f"Jan {intake_year} PT Intake"

        local_timetable_dir = os.path.join(
            "files",
            "MTech in EBAC_IS_SE (Thru-train)",
            "course files",
            "Timetable",
            mode_folder,
            intake_folder
        )
        if not os.path.exists(local_timetable_dir):
            return

        timetable_files = [
            os.path.join(local_timetable_dir, f)
            for f in os.listdir(local_timetable_dir)
            if f.lower().endswith(".pdf")
        ]
        if not timetable_files:
            return

        matched_file = self.fuzzy_match_file(timetable_files, course)
        if not matched_file:
            return

        timetable_info = self.extract_timetable_or_exam_date_from_local_file(matched_file, query)
        return timetable_info

    def extract_timetable_or_exam_date_from_local_file(self, file_path: str, query: str) -> str:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        full_text = ""
        for doc in docs:
            text = self.clean_text(doc.page_content)
            if len(text.split()) < self.OCR_WORD_THRESHOLD:
                ocr_texts = self.ocr_extract_text(file_path)
                if ocr_texts:
                    text = "\n".join([self.clean_text(t) for t in ocr_texts])
            full_text += text + "\n"
        if not full_text.strip():
            return None

        prompt = (
            f"Below is the timetable text extracted from the PDF:\n\n{full_text}\n\n"
            f"Based on this information, please answer the following query: {query}\n"
        )
        try:
            llm = ChatOpenAI(model_name="gpt-4.5-preview", temperature=0.5)
            response = llm.invoke(prompt)
            timetable_info = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return timetable_info
        except Exception:
            return None

    @staticmethod
    def fuzzy_match_file(files, course_code, threshold=0.5):
        for fpath in files:
            filename = os.path.basename(fpath)
            if filename.upper().startswith(course_code.upper()):
                return fpath

        best_score = 0.0
        best_file = None
        for fpath in files:
            filename = os.path.basename(fpath)
            ratio = difflib.SequenceMatcher(None, filename.upper(), course_code.upper()).ratio()
            if ratio > best_score:
                best_score = ratio
                best_file = fpath
        return best_file if best_score >= threshold else None

    # ======================================================
    # ASSIGNMENTS FUNCTIONS
    # ======================================================
    def get_assignments_due_dates(self, hide_older_than: int = 90, max_workers: int = None):
        if max_workers is None:
            max_workers = os.cpu_count() or 5
        assignments_list = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=hide_older_than)
        courses = list(self.canvas.get_courses())
        def process_course(course):
            local_assignments = []
            if getattr(course, "access_restricted_by_date", False):
                return local_assignments
            try:
                for assignment in course.get_assignments():
                    due_date_str = assignment.due_at
                    if not due_date_str:
                        continue
                    parsed_due_date = self._parse_date(due_date_str)
                    if parsed_due_date is not None and parsed_due_date < cutoff:
                        continue
                    local_assignments.append({
                        "course_name": course.name,
                        "assignment_name": assignment.name,
                        "due_at": parsed_due_date,
                        "due_at_str": due_date_str
                    })
            except Exception:
                pass
            return local_assignments
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_course, course): course for course in courses}
            for future in as_completed(futures):
                assignments_list.extend(future.result())
        sorted_assignments = sorted(
            assignments_list,
            key=lambda x: x["due_at"] if x["due_at"] is not None else datetime.max.replace(tzinfo=timezone.utc)
        )
        return sorted_assignments

    def list_upcoming_assignments(self, hide_older_than: int = 90) -> str:
        assignments = self.get_assignments_due_dates(hide_older_than=hide_older_than)
        result = []
        if assignments:
            result.append("Upcoming Assignments:")
            for idx, a in enumerate(assignments, start=1):
                result.append(f"{idx}. {a['due_at_str']}")
                result.append(f"    Course    : {a['course_name']}")
                result.append(f"    Assignment: {a['assignment_name']}")
        else:
            result.append("No upcoming assignments found.")
        return "\n".join(result)

    def get_assignment_detail(self, assignment_name: str, threshold: float = 0.8) -> str:
        courses = list(self.canvas.get_courses())
        matches = []
        for course in courses:
            if getattr(course, "access_restricted_by_date", False):
                continue
            try:
                for assignment in course.get_assignments():
                    score = self.get_match_score(assignment_name, assignment.name)
                    if score >= threshold:
                        matches.append((score, course, assignment))
            except Exception:
                pass
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            result = []
            result.append("Matching Assignments (showing due date and description):")
            for idx, (score, course, assignment) in enumerate(matches, start=1):
                desc = getattr(assignment, "description", None)
                if desc:
                    desc = self.strip_html(desc)
                result.append("-" * 50)
                result.append(f"Assignment {idx}:")
                result.append(f"  Course     : {course.name}")
                result.append(f"  Assignment : {assignment.name}")
                result.append(f"  Due Date   : {assignment.due_at}")
                result.append("  Description:")
                if desc:
                    for line in desc.splitlines():
                        result.append("    " + line)
                else:
                    result.append("    No description available")
                result.append(f"  Confidence : {score * 100:.1f}%")
                result.append("-" * 50)
            return "\n".join(result)
        else:
            return "Assignment not found."

    # ======================================================
    # ANNOUNCEMENTS FUNCTIONS
    # ======================================================
    def get_announcements(self, hide_older_than: int = 7, only_unread: bool = False, max_workers: int = None):
        if max_workers is None:
            max_workers = os.cpu_count() or 5
        announcements_list = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=hide_older_than)
        courses = list(self.canvas.get_courses())
        def process_course(course):
            local_announcements = []
            if getattr(course, "access_restricted_by_date", False):
                return local_announcements
            try:
                for ann in course.get_discussion_topics(only_announcements=True):
                    if only_unread and getattr(ann, "read_state", "unread") != "unread":
                        continue
                    created_at_str = getattr(ann, "created_at", None)
                    parsed_created_at = self._parse_date(created_at_str) if created_at_str else None
                    if parsed_created_at is not None and parsed_created_at < cutoff:
                        continue
                    local_announcements.append({
                        "course_name": course.name,
                        "announcement_title": ann.title,
                        "created_at": parsed_created_at,
                        "created_at_str": created_at_str if created_at_str else "No date"
                    })
            except Exception:
                pass
            return local_announcements
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_course, course): course for course in courses}
            for future in as_completed(futures):
                announcements_list.extend(future.result())
        sorted_announcements = sorted(
            announcements_list,
            key=lambda x: x["created_at"] if x["created_at"] is not None else datetime.max.replace(tzinfo=timezone.utc)
        )
        return sorted_announcements

    def list_announcements(self, hide_older_than: int = 7, only_unread: bool = False) -> str:
        announcements = self.get_announcements(hide_older_than=hide_older_than, only_unread=only_unread)
        result = []
        if announcements:
            result.append("Recent Announcements:")
            for idx, ann in enumerate(announcements, start=1):
                result.append(f"{idx}. Created at: {ann['created_at_str']}")
                result.append(f"    Course : {ann['course_name']}")
                result.append(f"    Title  : {ann['announcement_title']}")
        else:
            result.append("No announcements found.")
        return "\n".join(result)

    def get_announcement_detail(self, announcement_title: str, threshold: float = 0.8) -> str:
        courses = list(self.canvas.get_courses())
        matches = []
        for course in courses:
            if getattr(course, "access_restricted_by_date", False):
                continue
            try:
                for ann in course.get_discussion_topics(only_announcements=True):
                    score = self.get_match_score(announcement_title, ann.title)
                    if score >= threshold:
                        matches.append((score, course, ann))
            except Exception:
                pass
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            result = []
            result.append("Matching Announcements (showing created_at and description):")
            for idx, (score, course, ann) in enumerate(matches, start=1):
                desc = getattr(ann, "message", None) or getattr(ann, "description", None)
                if desc:
                    desc = CanvasManager.strip_html(desc)
                created_at = getattr(ann, "created_at", None)
                result.append("-" * 50)
                result.append(f"Announcement {idx}:")
                result.append(f"  Course     : {course.name}")
                result.append(f"  Title      : {ann.title}")
                result.append(f"  Created at : {created_at}")
                result.append("  Description:")
                if desc:
                    for line in desc.splitlines():
                        result.append("    " + line)
                else:
                    result.append("    No description available")
                result.append(f"  Confidence : {score * 100:.1f}%")
                result.append("-" * 50)
            return "\n".join(result)
        else:
            return "Announcement not found."


if __name__ == "__main__":
    try:
        with open("keys/canvas.txt", "r") as file:
            api_key = file.read().strip()
    except Exception:
        api_key = ""

    try:
        with open("keys/openai.txt", "r") as file:
            openai_key = file.read().strip()
    except Exception:
        openai_key = ""

    os.environ["OPENAI_API_KEY"] = openai_key

    API_URL = "https://canvas.nus.edu.sg/"
    manager = CanvasManager(API_URL, api_key)

    # List all available models
    list_all_models()

    # Uncomment to download files:
    # manager.download_all_files_parallel(base_dir="files")

    # Uncomment to build embedding index:
    # manager.build_embedding_index(index_dir="chroma_index")

    # Generate summaries; new summary text replaces old ones if incomplete.
    # manager.build_all_summaries(base_dir="files", summary_base_dir="summary")

    """
    # Example usage for retrieving lecture slides:
    results1 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["CNI"]
    )
    results2 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["UI"]
    )
    results3 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["EBA5004", "CUI"]
    )
    results4 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["TPML", "CUI"]
    )
    results5 = manager.retrieve_lecture_slides_by_topic(
        topic="what is transformer?",
        filter_terms=["VSD", "EBA5004"]
    )
    """
    # 2. TIMETABLE
    timetable_info = manager.get_timetable_or_exam_date(True, 2024, "AIS06", "I am AIS06 2024 FT batch, What is my exam date?")
    if timetable_info:
        print(timetable_info)

    # 3. ASSIGNMENTS & DEADLINES
    assignments_str = manager.list_upcoming_assignments(hide_older_than=0)
    print(assignments_str)
    assignment_details_str = manager.get_assignment_detail("CNI Day 4 Workshop")
    print(assignment_details_str)

    # 4. ANNOUNCEMENTS & NOTIFICATIONS
    announcements_str = manager.list_announcements(hide_older_than=7, only_unread=False)
    print(announcements_str)
    announcement_details_str = manager.get_announcement_detail("Internship Announcement")
    print(announcement_details_str)
