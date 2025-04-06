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
        print(f"[DEBUG] Running OCR on {pdf_path} ...")
        images = convert_from_path(pdf_path)
        texts = []
        for idx, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            print(f"[DEBUG] OCR extracted text for page {idx}:\n{text}\n")
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
    # DOWNLOAD FILES FUNCTIONS (unchanged)
    # ======================================================
    def download_all_files_parallel(self, base_dir: str = "files", max_workers: int = None):
        if max_workers is None:
            max_workers = os.cpu_count() or 4
        courses = list(self.canvas.get_courses())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for course in courses:
                if getattr(course, "access_restricted_by_date", False):
                    print(f"Skipping restricted course: {course.id}")
                    continue
                course_name = self.sanitize_filename(course.name)
                print(f"Processing course: {course_name}")
                try:
                    folder_path_map = self.build_folder_path_map(course)
                except Exception as e:
                    print(f"Error building folder path map for course {course.name}: {e}")
                    folder_path_map = {}
                try:
                    files = list(course.get_files())
                except Exception as e:
                    print(f"Error retrieving files for course {course.name}: {e}")
                    continue
                for f in files:
                    futures.append(executor.submit(self.download_file, f, course_name, folder_path_map, base_dir))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in a download task: {e}")

    def download_file(self, file_obj, course_name, folder_path_map, base_dir):
        try:
            folder_id = getattr(file_obj, "folder_id", None)
            folder_name = folder_path_map.get(folder_id, "")
            target_dir = os.path.join(base_dir, course_name, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, str(file_obj.display_name))
            print(f"Downloading {file_obj.display_name} to {file_path}...")
            file_obj.download(file_path)
        except Exception as e:
            print(f"Error downloading file {file_obj.display_name}: {e}")

    # ------------------------------------------------------
    # EMBEDDING INDEX BUILDING (with OCR fallback)
    # ------------------------------------------------------
    def build_embedding_index(self, index_dir: str = "chroma_index", base_dir: str = "files"):
        """
        Build the embedding index for all lecture slides by scanning the given base directory.
        Each PDF page is treated as a separate document; non-PDF files are split into chunks.
        Text from documents is cleaned before embeddings are computed.
        If a PDF's extracted text is too short (<OCR_WORD_THRESHOLD words), OCR is used as a fallback.
        """
        if os.path.exists(index_dir):
            print(f"[DEBUG] Clearing existing index directory: {index_dir}")
            shutil.rmtree(index_dir)
        print(f"\n[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Will store Chroma index in: {os.path.abspath(index_dir)}")
        print(f"[DEBUG] Processing all directories under base directory: {os.path.abspath(base_dir)}")
        cpu_cores = os.cpu_count() or 4
        print(f"[DEBUG] CPU cores available: {cpu_cores}")
        start_time = time.time()
        selected_paths = []
        if not os.path.exists(base_dir):
            print(f"Warning: Base directory '{base_dir}' does not exist.")
            return None
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.pptx', '.doc', '.docx')):
                    selected_paths.append(os.path.join(root, file))
        if not selected_paths:
            print("No files found in the base directory.")
            return None
        print(f"Found {len(selected_paths)} file(s) for building the index.\n")
        documents = []
        max_workers = os.cpu_count() or 4

        def load_file(fp):
            print(f"[DEBUG] Loading file: {fp}")
            try:
                docs = []
                if fp.lower().endswith('.pdf'):
                    loader = PyPDFLoader(fp)
                    docs = loader.load()
                    # If extracted text is too short (<OCR_WORD_THRESHOLD words), use OCR fallback.
                    for idx, doc in enumerate(docs):
                        if len(doc.page_content.split()) < CanvasManager.OCR_WORD_THRESHOLD:
                            print(f"[DEBUG] Page {idx+1} from {os.path.basename(fp)} has insufficient text. Running OCR fallback.")
                            ocr_texts = CanvasManager.ocr_extract_text(fp)
                            if idx < len(ocr_texts):
                                doc.page_content = ocr_texts[idx]
                else:
                    loader = UnstructuredLoader(fp)
                    docs = loader.load()
                if docs:
                    print(f"[DEBUG]   -> Loaded {len(docs)} document(s) from {os.path.basename(fp)}")
                for doc in docs:
                    if doc.page_content:
                        doc.page_content = CanvasManager.clean_text(doc.page_content)
                    doc.metadata["file_path"] = fp
                    doc.metadata["canvas_link"] = f"{self.api_url}/files/{os.path.basename(fp)}"
                return docs
            except Exception as e:
                print(f"Error loading file {fp}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_file, fp): fp for fp in selected_paths}
            for future in as_completed(futures):
                docs_from_file = future.result()
                documents.extend(docs_from_file)

        if not documents:
            print("No documents could be loaded from the selected files.")
            return None

        pdf_docs = []
        non_pdf_docs = []
        for doc in documents:
            if doc.metadata.get("file_path", "").lower().endswith(".pdf"):
                pdf_docs.append(doc)
            else:
                non_pdf_docs.append(doc)
        if non_pdf_docs:
            print(f"\n[DEBUG] Splitting {len(non_pdf_docs)} non-PDF document(s) into chunks...")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            non_pdf_docs = text_splitter.split_documents(non_pdf_docs)
            print(f"[DEBUG]   -> After splitting, we have {len(non_pdf_docs)} chunk(s) from non-PDF files.")
        docs = pdf_docs + non_pdf_docs

        new_docs = []
        for doc in docs:
            if isinstance(doc, tuple) or not hasattr(doc, "metadata"):
                try:
                    doc = Document(page_content=doc[0], metadata=doc[1])
                except Exception as e:
                    print("Skipping document due to conversion error:", e)
                    continue
            new_docs.extend(filter_complex_metadata([doc]))
        docs = new_docs

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("\nCUDA is available. Using GPU for embeddings.")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
        else:
            print("\nCUDA not available. Using CPU for embeddings.")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        try:
            print("\n[DEBUG] Building Chroma index from lecture slides...")
            vector_store = Chroma.from_documents(docs, embeddings, persist_directory=index_dir)
            vector_store.persist()
            print("[DEBUG] Embedding index built and persisted.")
            elapsed_time = time.time() - start_time
            print(f"[DEBUG] Index building took {elapsed_time:.2f} seconds.")
            return vector_store
        except Exception as e:
            print(f"Error building index: {e}")
            return None

    # ------------------------------------------------------
    # BUILD ALL SUMMARIES (using ChatGPT with OCR fallback if needed)
    # ------------------------------------------------------
    def build_all_summaries(self, base_dir: str = "files", summary_base_dir: str = "summary"):
        print(f"\n[DEBUG] Generating summaries for all files under: {os.path.abspath(base_dir)}")
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.pptx', '.doc', '.docx')):
                    file_path = os.path.join(root, file)
                    try:
                        CanvasManager.create_summary_csv(file_path, summary_base_dir, base_dir)
                    except Exception as e:
                        print(f"[DEBUG] Error generating summary for file {file_path}: {e}")

    # ------------------------------------------------------
    # Create Summary CSV File for a Given Document (Page by Page)
    # Using OCR fallback if extracted text < OCR_WORD_THRESHOLD words.
    # New summary text replaces old ones.
    # ------------------------------------------------------
    @classmethod
    def create_summary_csv(cls, file_path: str, summary_base_dir: str, base_dir: str):
        min_word_count = cls.OCR_WORD_THRESHOLD

        # Compute output CSV file path.
        rel_path = os.path.relpath(file_path, base_dir)
        summary_file_path = os.path.join(summary_base_dir, rel_path)
        summary_file_path = os.path.splitext(summary_file_path)[0] + ".csv"

        # Load the document.
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredLoader(file_path)
        try:
            docs = loader.load()
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return

        if not docs:
            print(f"[DEBUG] No documents loaded from {file_path}.")
            return

        os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)

        # Read existing summaries if file exists.
        summaries = []
        need_update = False
        if os.path.exists(summary_file_path):
            print(f"[DEBUG] Summary file {summary_file_path} exists. Verifying each row...")
            try:
                with open(summary_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_rows = list(reader)
            except Exception as e:
                print(f"Error reading existing summary CSV: {e}")
                existing_rows = []
            if len(existing_rows) != len(docs):
                print("[DEBUG] Mismatch in number of rows; regenerating all summaries.")
                need_update = True
            else:
                for row in existing_rows:
                    summary_text = row.get("Summary", "").strip()
                    if len(summary_text.split()) < min_word_count:
                        need_update = True
                        break
                if not need_update:
                    print("[DEBUG] All summaries appear complete. Skipping regeneration.")
                    return
                else:
                    print("[DEBUG] Some summaries are incomplete. Regenerating those entries.")
                    for row in existing_rows:
                        summaries.append(row.get("Summary", "").strip())

        # Pre-run OCR only once if any page has insufficient text
        ocr_texts = None
        if file_path.lower().endswith('.pdf'):
            insufficient = any(len(cls.clean_text(doc.page_content).split()) < min_word_count for doc in docs)
            if insufficient:
                print(f"[DEBUG] Detected insufficient text in at least one page. Running OCR fallback for {file_path}...")
                ocr_texts = cls.ocr_extract_text(file_path)

        # Loop over docs: update or generate summary.
        for i, doc in enumerate(docs, start=1):
            text = cls.clean_text(doc.page_content)
            print(f"\n[DEBUG] Extracted text for page {i} of {os.path.basename(file_path)}:\n{text}\n")
            # Use OCR text if extracted text is too short and OCR results are available.
            if file_path.lower().endswith('.pdf') and len(text.split()) < min_word_count and ocr_texts:
                if i - 1 < len(ocr_texts):
                    text = cls.clean_text(ocr_texts[i - 1])
                    print(f"[DEBUG] Using OCR extracted text for page {i}:\n{text}\n")
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
                        new_summary = cls.summarize_page_chatgpt(text, page_number=i,
                                                                 file_name=os.path.basename(file_path))
                        if len(new_summary.split()) >= min_word_count:
                            break
                        print(f"[DEBUG] Attempt {attempt} for page {i} produced a summary with fewer than {min_word_count} words; retrying...")
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
                    print(f"[DEBUG] Generated summary for page {i} of {file_path}.")
            print(f"[DEBUG] Summary CSV saved to: {summary_file_path}")
        except Exception as e:
            print(f"Error writing CSV for file {file_path}: {e}")

    # ------------------------------------------------------
    # Summarize a Page Using OpenAI ChatCompletion API
    # ------------------------------------------------------
    @staticmethod
    def summarize_page_chatgpt(text: str, page_number: int, file_name: str) -> str:
        prompt = (
            f"Please summarize the following content from page {page_number} of the document '{file_name}' "
            f"in a concise paragraph. Focus on the key points:\n\n{text}"
        )
        print(f"\n[DEBUG] Prompt for page {page_number}:\n{prompt}\n")
        try:
            llm = OpenAI(temperature=0.5, max_tokens=150)
            response = llm.invoke(prompt).strip()
            print(f"[DEBUG] Response from OpenAI for page {page_number}:\n{response}\n")
            return response
        except Exception as e:
            print(f"Error during summarization on page {page_number}: {e}")
            return "[API Error]"

    # ======================================================
    # RETRIEVE LECTURE SLIDES FUNCTIONS (and other functions remain unchanged)
    # ======================================================
    def retrieve_lecture_slides_by_topic(self, topic: str, index_dir: str = "chroma_index", k: int = 100,
                                           filter_terms: list = None):
        max_workers = os.cpu_count() or 4

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

        if filter_terms:
            print("Resolved course filter:", resolved_course)
            print("Resolved sub-module filter:", resolved_submodule)
        else:
            print("No filter terms provided; processing all courses and sub-modules.")

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
            print("No courses/sub-modules match the provided filter terms.")
            return []

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("CUDA is available. Using GPU for embeddings.")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
        else:
            print("CUDA not available. Using CPU for embeddings.")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        if os.path.exists(index_dir):
            try:
                print("Loading existing Chroma index...")
                vector_store = Chroma(persist_directory=index_dir, embedding_function=embeddings)
            except Exception as e:
                print(f"Error loading index: {e}")
                return []
        else:
            vector_store = self.build_embedding_index(index_dir=index_dir)
            if vector_store is None:
                return []

        try:
            results = vector_store.similarity_search(topic, k=k)
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
        print(f"Found {len(results)} relevant slide chunk(s) for topic: '{topic}'")

        if filter_terms:
            allowed_paths = []
            for course, submodules in courses_to_process.items():
                for submodule, allowed_path in submodules.items():
                    allowed_paths.append(allowed_path.lower())
            print("Allowed paths based on filter:", allowed_paths)
            filtered_results = []
            for res in results:
                file_path = res.metadata.get("file_path", "").lower()
                if any(allowed_path in file_path for allowed_path in allowed_paths):
                    filtered_results.append(res)
            results = filtered_results
            print(f"After filtering, found {len(results)} result(s) based on filter terms.")

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
            print(f"--- Result {idx} ---")
            print("Content:", res.page_content)
            print("File Path:", file_info)
            print("Filename:", filename)
            print("Page Info:", page_info)
            print("Canvas Link:", canvas_link)
            print("Additional Metadata:", res.metadata)
        return final_results

    # ======================================================
    # TIMETABLE AND EXAM DATE FUNCTIONS
    # ======================================================
    def get_timetable_or_exam_date(self, full_time: bool, intake_year: int, course: str, query: str):
        """
        Locates a local PDF file that matches the specified course code (direct or fuzzy match)
        in the correct timetable directory for full_time vs. part_time and the given intake_year.
        Then extracts the timetable information (including exam dates) using the Camelot table extraction method.
        """
        if full_time:
            mode_folder = "MTech Full Time"
            intake_folder = f"Aug {intake_year} FT Intake"
        else:
            mode_folder = "MTech Part Time"
            intake_folder = f"Jan {intake_year} PT Intake"

        # Build the local directory path.
        local_timetable_dir = os.path.join(
            "files",
            "MTech in EBAC_IS_SE (Thru-train)",
            "course files",
            "Timetable",
            mode_folder,
            intake_folder
        )
        print("[DEBUG] Local timetable directory:", local_timetable_dir)
        if not os.path.exists(local_timetable_dir):
            print(f"Local timetable directory not found: {local_timetable_dir}")
            return

        # List all files in the directory for debugging.
        print("[DEBUG] Files in directory:")
        for f in os.listdir(local_timetable_dir):
            print(" -", f)

        # Gather all PDF files in the local timetable directory.
        timetable_files = [
            os.path.join(local_timetable_dir, f)
            for f in os.listdir(local_timetable_dir)
            if f.lower().endswith(".pdf")
        ]
        if not timetable_files:
            print("No timetable PDF files found in the local directory.")
            return

        # Fuzzy match (or direct match) the user's course code to one of the PDF filenames.
        matched_file = self.fuzzy_match_file(timetable_files, course)
        if not matched_file:
            print(f"No matching timetable PDF found for course code '{course}'.")
            return

        print(f"Processing timetable file: {matched_file}")
        # Pass the query to the extraction function.
        timetable_info = self.extract_timetable_or_exam_date_from_local_file(matched_file, query)
        if timetable_info:
            print("\nExtracted Timetable Information:")
            print(timetable_info)
        else:
            print("Failed to extract timetable information.")

    def extract_timetable_or_exam_date_from_local_file(self, file_path: str, query: str) -> str:
        """
        Reads a local PDF file (which may contain images), applies OCR fallback if needed,
        and uses ChatGPT to extract and format the timetable information (including exam dates),
        based on the provided query.
        """
        print(f"[DEBUG] Reading timetable file from local path: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        full_text = ""
        for doc in docs:
            text = self.clean_text(doc.page_content)
            # If text is insufficient, try OCR.
            if len(text.split()) < self.OCR_WORD_THRESHOLD:
                print("[DEBUG] Insufficient text on a page, running OCR fallback for timetable...")
                ocr_texts = self.ocr_extract_text(file_path)
                if ocr_texts:
                    text = "\n".join([self.clean_text(t) for t in ocr_texts])
            full_text += text + "\n"
        if not full_text.strip():
            print("[DEBUG] No text extracted from the timetable file.")
            return None

        # Create a prompt that includes both the extracted text and the custom query.
        prompt = (
            f"Below is the timetable text extracted from the PDF:\n\n{full_text}\n\n"
            f"Based on this information, please answer the following query: {query}\n"
        )
        print(f"[DEBUG] Generated prompt for timetable extraction:\n{prompt}\n")
        try:
            # Use a powerful model without explicitly setting max_tokens.
            llm = ChatOpenAI(model_name="gpt-4.5-preview", temperature=0.5)
            response = llm.invoke(prompt)
            timetable_info = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return timetable_info
        except Exception as e:
            print("Error during ChatGPT extraction of timetable:", e)
            return None

    @staticmethod
    def fuzzy_match_file(files, course_code, threshold=0.5):
        """
        Given a list of file paths and a course code (e.g. 'AIS06'),
        first check if any file's name starts with the course code (case-insensitive).
        If so, return that file. Otherwise, return the file with the highest similarity ratio
        above the given threshold, or None if none qualify.
        """
        # First, try a direct startswith match.
        for fpath in files:
            filename = os.path.basename(fpath)
            if filename.upper().startswith(course_code.upper()):
                print(f"[DEBUG] Direct match found: {filename}")
                return fpath

        # Fallback to fuzzy matching.
        best_score = 0.0
        best_file = None
        print("[DEBUG] Starting fuzzy matching for course code:", course_code)
        for fpath in files:
            filename = os.path.basename(fpath)
            ratio = difflib.SequenceMatcher(None, filename.upper(), course_code.upper()).ratio()
            print(f"[DEBUG] Comparing file '{filename}' with course code '{course_code.upper()}': score = {ratio:.2f}")
            if ratio > best_score:
                best_score = ratio
                best_file = fpath
        print("[DEBUG] Best match score:", best_score)
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
            except Exception as e:
                print(f"Error retrieving assignments for course {course.name}: {e}")
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

    def list_upcoming_assignments(self, hide_older_than: int = 90):
        assignments = self.get_assignments_due_dates(hide_older_than=hide_older_than)
        if assignments:
            print("\nUpcoming Assignments:")
            for idx, a in enumerate(assignments, start=1):
                print(f"{idx}. {a['due_at_str']}")
                print(f"    Course    : {a['course_name']}")
                print(f"    Assignment: {a['assignment_name']}")
        else:
            print("No upcoming assignments found.")
        return assignments

    def get_assignment_detail(self, assignment_name: str, threshold: float = 0.8):
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
            except Exception as e:
                print(f"Error retrieving assignments for course {course.name}: {e}")
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            print("\nMatching Assignments (showing due_at and description):")
            results = []
            for idx, (score, course, assignment) in enumerate(matches, start=1):
                desc = getattr(assignment, "description", None)
                if desc:
                    desc = self.strip_html(desc)
                print("-" * 50)
                print(f"Assignment {idx}:")
                print(f"  Course     : {course.name}")
                print(f"  Assignment : {assignment.name}")
                print(f"  Due Date   : {assignment.due_at}")
                print("  Description:")
                if desc:
                    for line in desc.splitlines():
                        print("    " + line)
                else:
                    print("    No description available")
                print(f"  Confidence : {score * 100:.1f}%")
                print("-" * 50)
                results.append({"due_at": assignment.due_at, "description": desc})
            return results
        else:
            print("Assignment not found.")
            return []

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
            except Exception as e:
                print(f"Error retrieving announcements for course {course.name}: {e}")
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

    def list_announcements(self, hide_older_than: int = 7, only_unread: bool = False):
        announcements = self.get_announcements(hide_older_than=hide_older_than, only_unread=only_unread)
        if announcements:
            print("\nRecent Announcements:")
            for idx, ann in enumerate(announcements, start=1):
                print(f"{idx}. Created at: {ann['created_at_str']}")
                print(f"    Course : {ann['course_name']}")
                print(f"    Title  : {ann['announcement_title']}")
        else:
            print("No announcements found.")
        return announcements

    def get_announcement_detail(self, announcement_title: str, threshold: float = 0.8):
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
            except Exception as e:
                print(f"Error retrieving announcements for course {course.name}: {e}")
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            print("\nMatching Announcements (showing created_at and description):")
            results = []
            for idx, (score, course, ann) in enumerate(matches, start=1):
                desc = getattr(ann, "message", None) or getattr(ann, "description", None)
                if desc:
                    desc = CanvasManager.strip_html(desc)
                created_at = getattr(ann, "created_at", None)
                print("-" * 50)
                print(f"Announcement {idx}:")
                print(f"  Course     : {course.name}")
                print(f"  Title      : {ann.title}")
                print(f"  Created at : {created_at}")
                print("  Description:")
                if desc:
                    for line in desc.splitlines():
                        print("    " + line)
                else:
                    print("    No description available")
                print(f"  Confidence : {score * 100:.1f}%")
                print("-" * 50)
                results.append({
                    "created_at": created_at,
                    "description": desc,
                    "full_details": ann.__dict__
                })
            return results
        else:
            print("Announcement not found.")
            return []


if __name__ == "__main__":
    try:
        with open("keys/canvas.txt", "r") as file:
            api_key = file.read().strip()
    except Exception as e:
        print("Error reading Canvas API key:", e)
        api_key = ""

    try:
        with open("keys/openai.txt", "r") as file:
            openai_key = file.read().strip()
    except Exception as e:
        print("Error reading OpenAI API key:", e)
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
    print("\nExample 1: Using alias filter term 'CNI'")
    results1 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["CNI"]
    )
    print("\nFinal top results from Example 1:", results1)

    print("\nExample 2: Using filter term 'UI'")
    results2 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["UI"]
    )
    print("\nFinal top results from Example 2:", results2)

    print("\nExample 3: Using filter terms 'EBA5004' and 'CUI'")
    results3 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["EBA5004", "CUI"]
    )
    print("\nFinal top results from Example 3:", results3)

    print("\nExample 4: Using filter terms 'TPML' and 'CUI'")
    results4 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?",
        filter_terms=["TPML", "CUI"]
    )
    print("\nFinal top results from Example 4:", results4)

    print("\nExample 5: Using filter terms 'VSD' and 'EBA5004'")
    results5 = manager.retrieve_lecture_slides_by_topic(
        topic="what is transformer?",
        filter_terms=["VSD", "EBA5004"]
    )
    print("\nFinal top results from Example 5:", results5)
    """
    # 2. TIMETABLE
    manager.get_timetable_or_exam_date(True, 2024, "AIS06", "I am AIS06 2024 FT batch, What is my exam date?")

    # 3. ASSIGNMENTS & DEADLINES
    # manager.list_upcoming_assignments(hide_older_than=0)
    # manager.get_assignment_detail("CNI Day 4 Workshop")

    # 4. ANNOUNCEMENTS & NOTIFICATIONS
    # manager.list_announcements(hide_older_than=7, only_unread=False)
    # manager.get_announcement_detail("Internship Announcement")
