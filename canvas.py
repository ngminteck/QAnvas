import os
import re
import difflib
import time
import html
import csv
import shutil
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import requests
import tiktoken
from canvasapi import Canvas

# Embedding, document loaders, and vector store libraries.
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from langchain_openai import OpenAI, ChatOpenAI

# OCR extraction packages.
from pdf2image import convert_from_path
import pytesseract


def list_all_models():
    """
    Print all available OpenAI models.
    Requires the OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
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
    CanvasManager handles operations related to a Canvas learning management system.
    Operations include retrieving lecture slides, downloading files,
    building embedding indexes, summarizing documents, retrieving timetables,
    assignments, and announcements.

    Adjustable OCR fallback threshold is available as a class attribute.
    """
    OCR_WORD_THRESHOLD = 5

    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the Canvas Manager.

        Parameters:
            api_url (str): The base URL for the Canvas instance.
            api_key (str): The API key for authentication.
        """
        self.canvas = Canvas(api_url, api_key)
        self.api_url = api_url

    # ----------------------------------------------------------------------------
    # Helper / Utility Functions
    # ----------------------------------------------------------------------------
    @staticmethod
    def get_match_score(query: str, title: str) -> float:
        """
        Compute a normalized match score between a query and title.
        """
        query_norm = re.sub(r'\W+', ' ', query).strip().lower()
        title_norm = re.sub(r'\W+', ' ', title).strip().lower()
        if query_norm in title_norm:
            return 1.0
        return difflib.SequenceMatcher(None, query_norm, title_norm).ratio()

    @staticmethod
    def strip_html(text: str) -> str:
        """
        Remove HTML tags and unescape HTML entities.
        """
        text = re.sub(r'<[^>]+>', '', text)
        return html.unescape(text)

    @staticmethod
    def _parse_date(date_str: str):
        """
        Parse a Canvas date string into a datetime object with UTC timezone.
        """
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            except Exception:
                return None

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Sanitize filenames by replacing forbidden characters.
        """
        return re.sub(r'[\\/*?:"<>|]', "_", name)

    @classmethod
    def build_folder_path_map(cls, course) -> dict:
        """
        Build a mapping from folder ID to its full path (folder hierarchy) for a course.
        """
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
    def ocr_extract_text(pdf_path: str) -> list:
        """
        Extract text from a PDF file using OCR.
        """
        images = convert_from_path(pdf_path)
        texts = [pytesseract.image_to_string(image) for image in images]
        return texts

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and sanitize text (strip HTML, remove sensitive info, collapse whitespace).
        """
        text = CanvasManager.strip_html(text)
        text = CanvasManager.remove_sensitive_info(text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_sensitive_info(text: str) -> str:
        """
        Replace email addresses and phone numbers with redacted markers.
        """
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        return re.sub(r'(\+?\d[\d\-\s().]{7,}\d)', '[REDACTED_PHONE]', text)

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in the text using the tiktoken package.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    # ----------------------------------------------------------------------------
    # File Download Functions
    # ----------------------------------------------------------------------------
    def download_all_files_parallel(self, base_dir: str = "files", max_workers: int = None):
        """
        Download all files in parallel from all available Canvas courses.
        """
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
                for file_obj in files:
                    futures.append(
                        executor.submit(self.download_file, file_obj, course_name, folder_path_map, base_dir))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

    def download_file(self, file_obj, course_name, folder_path_map, base_dir):
        """
        Download a single file from Canvas.
        """
        try:
            folder_id = getattr(file_obj, "folder_id", None)
            folder_name = folder_path_map.get(folder_id, "")
            target_dir = os.path.join(base_dir, course_name, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, str(file_obj.display_name))
            file_obj.download(file_path)
        except Exception:
            pass

    # ----------------------------------------------------------------------------
    # Embedding Index & Summaries
    # ----------------------------------------------------------------------------
    def build_embedding_index(self, index_dir: str = "chroma_index", base_dir: str = "files"):
        """
        Build an embedding index (using Chroma) for documents in the specified base directory.
        Uses OCR fallback for PDF files with insufficient text.
        """
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        start_time = time.time()
        selected_paths = []
        if not os.path.exists(base_dir):
            return None
        for root, _, files in os.walk(base_dir):
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
                    # Attempt to extract tables using pdfplumber and append table text.
                    import pdfplumber
                    try:
                        with pdfplumber.open(fp) as pdf:
                            table_texts = []
                            for page in pdf.pages:
                                tables = page.extract_tables()
                                for table in tables:
                                    formatted_rows = ["\t".join(cell if cell is not None else "" for cell in row)
                                                      for row in table]
                                    if formatted_rows:
                                        table_texts.append("\n".join(formatted_rows))
                            if table_texts:
                                table_text_block = "\n\n" + "\n\n".join(table_texts)
                                for doc in docs:
                                    doc.page_content += table_text_block
                    except Exception:
                        pass
                    # Use OCR fallback if the extracted text is too short.
                    for idx, doc in enumerate(docs):
                        if len(doc.page_content.split()) < CanvasManager.OCR_WORD_THRESHOLD:
                            ocr_texts = CanvasManager.ocr_extract_text(fp)
                            if idx < len(ocr_texts):
                                doc.page_content = ocr_texts[idx]
                else:
                    loader = UnstructuredLoader(fp)
                    docs = loader.load()
                # Clean document text and add metadata.
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

        pdf_docs, non_pdf_docs = [], []
        for doc in documents:
            if doc.metadata.get("file_path", "").lower().endswith(".pdf"):
                pdf_docs.append(doc)
            else:
                non_pdf_docs.append(doc)
        if non_pdf_docs:
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            non_pdf_docs = splitter.split_documents(non_pdf_docs)
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})

        try:
            vector_store = Chroma.from_documents(docs, embeddings, persist_directory=index_dir)
            vector_store.persist()
            elapsed_time = time.time() - start_time
            return vector_store
        except Exception:
            return None

    def build_all_summaries(self, base_dir: str = "files", summary_base_dir: str = "summary"):
        """
        Generate summary CSV files for each document in the base directory.
        Existing summaries are replaced if incomplete.
        """
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.pptx', '.doc', '.docx')):
                    file_path = os.path.join(root, file)
                    try:
                        CanvasManager.create_summary_csv(file_path, summary_base_dir, base_dir)
                    except Exception:
                        pass

    @classmethod
    def create_summary_csv(cls, file_path: str, summary_base_dir: str, base_dir: str):
        """
        Create a CSV file summarizing each page of the document.
        OCR fallback is used if the text is insufficient.
        """
        min_word_count = cls.OCR_WORD_THRESHOLD
        rel_path = os.path.relpath(file_path, base_dir)
        summary_file_path = os.path.join(summary_base_dir, rel_path)
        summary_file_path = os.path.splitext(summary_file_path)[0] + ".csv"

        loader = PyPDFLoader(file_path) if file_path.lower().endswith('.pdf') else UnstructuredLoader(file_path)
        try:
            docs = loader.load()
        except Exception:
            return
        if not docs:
            return

        os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
        summaries, need_update = [], False
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
                if need_update:
                    for row in existing_rows:
                        summaries.append(row.get("Summary", "").strip())
                else:
                    return

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

    @staticmethod
    def summarize_page_chatgpt(text: str, page_number: int, file_name: str) -> str:
        """
        Summarize the provided text for a specific page using OpenAI's ChatCompletion.
        """
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

    # ----------------------------------------------------------------------------
    # Lecture Slide Retrieval
    # ----------------------------------------------------------------------------
    def retrieve_lecture_slides_by_topic(self, query: str, subjects: list = None) -> str:
        """
        Retrieve relevant lecture slides based on the query and an optional list of subjects.
        Uses similarity search on an embedding index.
        """
        index_dir = "chroma_index"
        k = 100

        def resolve_filter(filter_list, alias_mapping, threshold=0.8):
            resolved = []
            for item in filter_list:
                item_lower = item.lower()
                for canonical, aliases in alias_mapping.items():
                    if CanvasManager.get_match_score(item_lower, canonical.lower()) >= threshold:
                        resolved.append(canonical)
                        break
                    elif any(
                            CanvasManager.get_match_score(item_lower, alias.lower()) >= threshold for alias in aliases):
                        resolved.append(canonical)
                        break
            return resolved

        # Define course and sub-module alias mappings.
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

        resolved_course = resolve_filter(subjects, course_aliases, threshold=0.8) if subjects else []
        resolved_submodule = resolve_filter(subjects, sub_module_aliases, threshold=0.8) if subjects else []

        file_paths = {
            "ISY5004": {
                "VSD": {"path": r"files\course files\Vision Systems (6-10Jan 2025)"},
                "SRSD": {"path": r"files\course files\Spatial Reasoning from Sensor Data (13-15Jan 2025)"},
                "RTAVS": {"path": r"files\course files\Real-time Audio-Visual Sensing and Sense Making (20-23Jan 2025)"}
            },
            "EBA5004": {
                "TA": {"path": r"files\course files\[PLP] Text Analytics (2025-02-10)"},
                "NMSM": {"path": r"files\course files\EBA5004 Practical Language Processing [2420]\01_NMSM"},
                "TPML": {"path": r"files\course files\EBA5004 Practical Language Processing [2420]\02_TPML"},
                "CUI": {"path": r"files\course files\EBA5004 Practical Language Processing [2420]\03_CNI"}
            }
        }

        courses_to_process = {}
        for course, submodules in file_paths.items():
            matching_submodules = {sm: details["path"] for sm, details in submodules.items() if
                                   sm in resolved_submodule}
            if matching_submodules:
                courses_to_process[course] = matching_submodules
            elif course in resolved_course:
                courses_to_process[course] = {sm: details["path"] for sm, details in submodules.items()}

        if not subjects:
            courses_to_process = {course: {sm: details["path"] for sm, details in submodules.items()}
                                  for course, submodules in file_paths.items()}

        if not courses_to_process:
            return "No course/sub-module matches provided filter terms."

        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})

        if os.path.exists(index_dir):
            try:
                vector_store = Chroma(persist_directory=index_dir, embedding_function=embeddings)
            except Exception as e:
                return f"Error loading vector store: {str(e)}"
        else:
            vector_store = self.build_embedding_index(index_dir=index_dir)
            if vector_store is None:
                return "Error building vector store."

        try:
            results = vector_store.similarity_search(query, k=k)
        except Exception as e:
            return f"Error during similarity search: {str(e)}"

        if subjects:
            allowed_paths = []
            for course, submodules in courses_to_process.items():
                for submodule, allowed_path in submodules.items():
                    allowed_paths.append(allowed_path.lower())
            filtered_results = [res for res in results if any(allowed_path in res.metadata.get("file_path", "").lower()
                                                              for allowed_path in allowed_paths)]
            results = filtered_results

        if not results:
            return "No relevant lecture slide excerpts were found."

        # Combine excerpts with metadata using dynamic token count.
        max_tokens = 8000
        excerpts_references = ""
        tokens_used = 0

        for res in results:
            file_info = res.metadata.get("file_path", "Unknown file")
            page_info = res.metadata.get("page", "N/A")
            canvas_link = res.metadata.get("canvas_link", "Not available")
            excerpt = res.page_content
            block = (f"File: {file_info}\nPage: {page_info}\nLink: {canvas_link}\nExcerpt:\n{excerpt}\n---\n")
            block_tokens = CanvasManager.count_tokens(block)
            if tokens_used + block_tokens > max_tokens:
                break
            excerpts_references += block
            tokens_used += block_tokens

        print(excerpts_references)

        synthesis_prompt = (
            "You are an expert lecturer assistant. Below are content from lecture slide documents, each with its reference metadata "
            "(including file, page, and link). Analyze the content and provide a detailed, clear answer to the query. In your answer, "
            "include a reference section that cites the source details for each key piece of information.\n\n"
            f"Content and References:\n{excerpts_references}\nQuery: {query}\n"
        )

        try:
            llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
            response = llm.invoke(synthesis_prompt)
            final_answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return final_answer
        except Exception as e:
            return f"Error during final answer generation: {str(e)}"

    # ----------------------------------------------------------------------------
    # Timetable & Exam Date Functions
    # ----------------------------------------------------------------------------
    def get_timetable_or_exam_date(self, full_time: bool, intake_year: int, course: str):
        """
        Retrieve timetable or exam schedule for a specified course.
        """
        if full_time:
            mode_folder = "MTech Full Time"
            intake_folder = f"Aug {intake_year} FT Intake"
        else:
            mode_folder = "MTech Part Time"
            intake_folder = f"Jan {intake_year} PT Intake"

        local_timetable_dir = os.path.join("files", "MTech in EBAC_IS_SE (Thru-train)", "course files", "Timetable",
                                           mode_folder, intake_folder)
        if not os.path.exists(local_timetable_dir):
            return

        timetable_files = [os.path.join(local_timetable_dir, f) for f in os.listdir(local_timetable_dir)
                           if f.lower().endswith(".pdf")]
        if not timetable_files:
            return

        matched_file = self.fuzzy_match_file(timetable_files, course)
        if not matched_file:
            return

        return self.extract_timetable_or_exam_date_from_local_file(matched_file)

    def extract_timetable_or_exam_date_from_local_file(self, file_path: str) -> str:
        """
        Extract and return timetable information (from tables) from a local PDF file.
        """
        import pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                table_texts = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        formatted_rows = ["\t".join(cell if cell is not None else "" for cell in row) for row in table]
                        if formatted_rows:
                            table_texts.append("\n".join(formatted_rows))
                full_text = "\n\n".join(table_texts)
            if not full_text.strip():
                return "Error: No table content found in the PDF."
        except Exception as e:
            return f"Error during PDF table extraction: {str(e)}"
        return full_text

    @staticmethod
    def fuzzy_match_file(files, course_code, threshold=0.5):
        """
        Fuzzy-match a course code to available file names and return the best match.
        """
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

    # ----------------------------------------------------------------------------
    # Assignments Functions
    # ----------------------------------------------------------------------------
    def get_assignments_due_dates(self):
        """
        Get a list of upcoming assignments (filtering out assignments older than 90 days).
        """
        hide_older_than = 90
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=hide_older_than)
        assignments_list = []
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

        max_workers = os.cpu_count() or 5
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_course, course): course for course in courses}
            for future in as_completed(futures):
                assignments_list.extend(future.result())

        sorted_assignments = sorted(assignments_list,
                                    key=lambda x: x["due_at"] if x["due_at"] is not None else datetime.max.replace(
                                        tzinfo=timezone.utc))
        return sorted_assignments

    def list_upcoming_assignments(self) -> str:
        """
        Return a formatted string of upcoming assignments.
        """
        assignments = self.get_assignments_due_dates()
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

    def get_assignment_detail(self, assignment_name: str) -> str:
        """
        Retrieve assignment details by matching an assignment name.
        """
        threshold = 0.8
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
            result = ["Matching Assignments (showing due date and description):"]
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

    # ----------------------------------------------------------------------------
    # Announcements Functions
    # ----------------------------------------------------------------------------
    def get_announcements(self):
        """
        Retrieve recent announcements from courses, filtering out those older than 7 days.
        """
        hide_older_than = 7
        only_unread = False
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=hide_older_than)
        announcements_list = []
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

        max_workers = os.cpu_count() or 5
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_course, course): course for course in courses}
            for future in as_completed(futures):
                announcements_list.extend(future.result())

        sorted_announcements = sorted(announcements_list,
                                      key=lambda x: x["created_at"] if x[
                                                                           "created_at"] is not None else datetime.max.replace(
                                          tzinfo=timezone.utc))
        return sorted_announcements

    def list_announcements(self) -> str:
        """
        Return a formatted string listing recent announcements.
        """
        announcements = self.get_announcements()
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

    def get_announcement_detail(self, announcement_title: str) -> str:
        """
        Retrieve and display detailed information about a specific announcement.
        """
        threshold = 0.8
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
            result = ["Matching Announcements (showing created_at and description):"]
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

    # Uncomment to list all available OpenAI models.
    # list_all_models()

    # Uncomment to download all files:
    # manager.download_all_files_parallel(base_dir="files")

    # Uncomment to build the embedding index:
    # manager.build_embedding_index(index_dir="chroma_index")

    # Uncomment to build summaries:
    # manager.build_all_summaries(base_dir="files", summary_base_dir="summary")

    # Example usage for retrieving lecture slides:
    results1 = manager.retrieve_lecture_slides_by_topic(query="how to implement langchain?")
    print(results1)

    # Timetable
    timetable_info = manager.get_timetable_or_exam_date(True, 2024, "AIS06")
    if timetable_info:
        print(timetable_info)

    # Assignments & Deadlines
    assignments_str = manager.list_upcoming_assignments()
    print(assignments_str)
    assignment_details_str = manager.get_assignment_detail("CNI Day 4 Workshop")
    print(assignment_details_str)

    # Announcements & Notifications
    announcements_str = manager.list_announcements()
    print(announcements_str)
    announcement_details_str = manager.get_announcement_detail("Internship Announcement")
    print(announcement_details_str)
