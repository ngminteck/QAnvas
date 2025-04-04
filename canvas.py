import os
import re
import difflib
import time
import html
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from transformers import pipeline
import shutil

import torch
from canvasapi import Canvas

# Third-party libraries
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document


class CanvasManager:
    """
    A manager class to handle Canvas operations such as retrieving lecture slides,
    downloading files, managing timetables, assignments, announcements, etc.
    """

    # Initialize the summarization pipeline with T5 (for summarization)
    _summarizer = pipeline(
        "summarization",
        model="t5-small",   # you can change to a different model if desired
        tokenizer="t5-small"
    )

    def __init__(self, api_url: str, api_key: str):
        self.canvas = Canvas(api_url, api_key)
        self.api_url = api_url

    # ------------------------------------------------------
    # Helper: Clean Text (Pre-Processing)
    # ------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean the input text by stripping HTML tags, removing sensitive personal information,
        normalizing whitespace, and removing extraneous characters.
        """
        # Remove HTML tags and decode entities.
        text = CanvasManager.strip_html(text)
        # Remove sensitive personal information (emails and phone numbers).
        text = CanvasManager.remove_sensitive_info(text)
        # Replace multiple whitespace with a single space.
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def remove_sensitive_info(text: str) -> str:
        """
        Remove sensitive personal information such as email addresses and phone numbers.
        """
        # Remove email addresses.
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        # Remove phone numbers: matches various formats including international ones.
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
    # EMBEDDING INDEX BUILDING (with cleaning)
    # ------------------------------------------------------
    def build_embedding_index(self, index_dir: str = "chroma_index", base_dir: str = "files"):
        """
        Build the embedding index for all lecture slides by scanning the given base directory.
        Each PDF page is treated as a separate document; non-PDF files are split into chunks.
        Text from documents is cleaned before embeddings are computed.
        This method does NOT generate summary CSV files.
        Before building the new index, the existing index directory is removed (if it exists)
        to ensure a complete rebuild.
        """
        # Clear existing index directory
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
                if fp.lower().endswith('.pdf'):
                    loader = PyPDFLoader(fp)
                    docs = loader.load()
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
    # BUILD ALL SUMMARIES (separate method)
    # ------------------------------------------------------
    def build_all_summaries(self, base_dir: str = "files", summary_base_dir: str = "summary"):
        """
        Traverse the base directory and generate summary CSV files for every supported file.
        The folder structure is preserved.
        """
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
    # SUMMARY HELPER FUNCTIONS (with cleaning and post-processing)
    # ------------------------------------------------------
    @staticmethod
    def generate_summary(text: str, max_length: int = 50, min_length: int = 25) -> str:
        text = CanvasManager.clean_text(text)
        tokens = CanvasManager._summarizer.tokenizer(text, return_tensors="pt", truncation=True)["input_ids"]
        input_length = tokens.shape[1]
        if input_length <= min_length:
            return text
        adjusted_max_length = max_length if input_length >= max_length else input_length
        result = CanvasManager._summarizer(
            text,
            max_length=adjusted_max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        summary = result[0]['summary_text'] if isinstance(result, list) else result['summary_text']
        # Post-process summary to clean extra spaces.
        return CanvasManager.clean_text(summary)

    @staticmethod
    def create_summary_csv(file_path: str, summary_base_dir: str = "summary", source_base_dir: str = "files") -> None:
        print(f"[DEBUG] Generating summary CSV for file: {file_path}")
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredLoader(file_path)
        docs = loader.load()
        if not file_path.lower().endswith('.pdf'):
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(docs)
        texts = [CanvasManager.clean_text(doc.page_content) for doc in docs if doc.page_content.strip()]
        results = [None] * len(texts)
        indices_to_summarize = []
        texts_to_summarize = []
        for i, text in enumerate(texts):
            tokens = CanvasManager._summarizer.tokenizer(text, return_tensors="pt", truncation=True)["input_ids"]
            if tokens.shape[1] < 10:
                results[i] = text
            else:
                indices_to_summarize.append(i)
                texts_to_summarize.append(text)
        if texts_to_summarize:
            batch_results = CanvasManager._summarizer(
                texts_to_summarize,
                max_length=50,
                min_length=25,
                do_sample=False,
                truncation=True
            )
            for idx, res in zip(indices_to_summarize, batch_results):
                results[idx] = CanvasManager.clean_text(res['summary_text'])
        page_summaries = [(f"Page {i + 1}", summary) for i, summary in enumerate(results)]
        overall_text = " ".join(texts)
        # Generate a longer overall summary
        overall_summary = CanvasManager.generate_summary(overall_text, max_length=150, min_length=50)
        print("[DEBUG] Generated overall summary")
        relative_path = os.path.relpath(file_path, source_base_dir)
        csv_relative_path = os.path.splitext(relative_path)[0] + ".csv"
        csv_output_path = os.path.join(summary_base_dir, csv_relative_path)
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Page", "Summary"])
            for page, summary in page_summaries:
                writer.writerow([page, summary])
            writer.writerow(["Overall Summary", overall_summary])
        print(f"[DEBUG] Summary CSV saved at {csv_output_path}")

    # ======================================================
    # RETRIEVE LECTURE SLIDES FUNCTIONS
    # ======================================================
    def retrieve_lecture_slides_by_topic(self, topic: str, index_dir: str = "chroma_index", k: int = 10,
                                           filter_terms: list = None):
        """
        Retrieve lecture slides related to a given topic using semantic search with Chroma.
        This method first checks for an existing precomputed index; if not found, it builds
        the index (i.e. precomputes all embeddings) before performing the search.
        """
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

        # --- Additional Filtering Step Based on Filter Terms ---
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
        # --- End Filtering Step ---

        top_results = results[:k]
        final_results = []
        for idx, res in enumerate(top_results, start=1):
            file_info = res.metadata.get("file_path", "Unknown file")
            page_info = res.metadata.get("page", "N/A")
            canvas_link = res.metadata.get("canvas_link", "Not available")
            filename = os.path.basename(file_info) if file_info != "Unknown file" else file_info
            final_results.append({
                "result_rank": idx,
                "content_preview": res.page_content,
                "file_path": file_info,
                "filename": filename,
                "page": page_info,
                "canvas_link": canvas_link,
                "metadata": res.metadata
            })
            print(f"--- Result {idx} ---")
            preview = res.page_content[:200] + "..." if len(res.page_content) > 200 else res.page_content
            print("Content Preview:", preview)
            print("File Path:", file_info)
            print("Filename:", filename)
            print("Page Info:", page_info)
            print("Canvas Link:", canvas_link)
            print("Additional Metadata:", res.metadata)

        return final_results

    # ======================================================
    # TIMETABLE FUNCTIONS
    # ======================================================
    def get_timetable(self, full_time: bool, intake: int, course: str):
        """
        Retrieve and display timetable files for the specified parameters.
        The expected folder structure is:
          course files/Timetable/MTech Full Time/Aug <intake> FT Intake  (for full-time)
          course files/Timetable/MTech Part Time/Jan <intake> PT Intake  (for part-time)
        Then, using the course code, only the matching timetable file(s) are shown.
        """
        if full_time:
            mode_folder = "MTech Full Time"
            intake_folder = f"Aug {intake} FT Intake"
        else:
            mode_folder = "MTech Part Time"
            intake_folder = f"Jan {intake} PT Intake"

        target_course_name = "MTech in EBAC/IS/SE (Thru-train)".lower()

        try:
            courses = list(self.canvas.get_courses())
        except Exception as e:
            print("Error retrieving courses:", e)
            return

        found_course = next(
            (c for c in courses if not getattr(c, "access_restricted_by_date", False)
             and target_course_name in c.name.lower()),
            None
        )

        if found_course is None:
            print(f"Timetable course '{target_course_name}' not found.")
            return

        print("Found timetable course:", found_course.name)

        try:
            files = list(found_course.get_files())
        except Exception as e:
            print("Error retrieving files from timetable course:", e)
            return

        try:
            folder_map = self.build_folder_path_map(found_course)
        except Exception as e:
            print("Error building folder mapping:", e)
            folder_map = {}

        target_folder = f"course files/Timetable/{mode_folder}/{intake_folder}"
        print(f"Looking for files in folder: {target_folder}")

        files_in_target = []
        for f in files:
            folder_id = getattr(f, "folder_id", None)
            folder_name = folder_map.get(folder_id, None)
            if folder_name and folder_name.lower() == target_folder.lower():
                files_in_target.append(f.display_name)

        if not files_in_target:
            print(f"No files found in folder {target_folder}.")
            return

        course_code = course.upper()
        files_for_course = [fname for fname in files_in_target if course_code in fname.upper()]

        if files_for_course:
            print(f"\nFiles for course code {course}:")
            for file_name in files_for_course:
                print("  -", file_name)
        else:
            print(f"No timetable file found for course code {course}.")

    # ======================================================
    # ASSIGNMENTS FUNCTIONS
    # ======================================================
    def get_assignments_due_dates(self, hide_older_than: int = 90, max_workers: int = None):
        """
        Retrieve assignments from all accessible courses concurrently that have a due date,
        skipping assignments without a due date or those due before (now - hide_older_than days).
        Returns a list of assignments sorted by due date.
        """
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
        """
        List upcoming assignments based on due dates.
        """
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
        """
        Retrieve detailed information for assignments that match the given name.
        Uses fuzzy matching to compute a confidence score for each assignment.
        All assignments with a confidence score equal to or above the threshold are printed,
        showing only the 'due_at' and 'description' fields (with HTML tags removed).
        """
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
        """
        Retrieve announcements from all accessible courses concurrently and sort them by creation date.
        Announcements older than (now - hide_older_than days) are skipped. If only_unread is True,
        only include announcements marked as unread.
        """
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
        """
        List recent announcements.
        """
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
        """
        Retrieve detailed content for announcements that match the given title.
        Uses fuzzy matching to compute a confidence score and prints all announcements
        with a score equal to or above the threshold.
        Only the 'created_at' and the description (from message or description, with HTML removed)
        fields are shown.
        """
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

    # ------------------------------------------------------
    # HELPER / UTILITY FUNCTIONS
    # ------------------------------------------------------
    @staticmethod
    def get_match_score(query: str, title: str) -> float:
        """
        Normalize both strings and compute a similarity ratio using difflib.
        If the normalized query is found as a substring in the normalized title,
        returns 1.0 (i.e. 100% confidence).
        """
        query_norm = re.sub(r'\W+', ' ', query).strip().lower()
        title_norm = re.sub(r'\W+', ' ', title).strip().lower()
        if query_norm in title_norm:
            return 1.0
        return difflib.SequenceMatcher(None, query_norm, title_norm).ratio()

    @staticmethod
    def strip_html(text: str) -> str:
        """
        Remove HTML tags from the given text and decode HTML entities.
        """
        text = re.sub(r'<[^>]+>', '', text)
        return html.unescape(text)

    @staticmethod
    def _parse_date(date_str: str):
        """
        Attempt to parse an ISO formatted date string.
        Supports formats with or without fractional seconds.
        """
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
        """
        Replace characters not allowed in filenames with an underscore.
        """
        return re.sub(r'[\\/*?:"<>|]', "_", name)

    @classmethod
    def build_folder_path_map(cls, course) -> dict:
        """
        Build a mapping of folder_id to full folder path (relative to the course root)
        using the folders returned by course.get_folders().
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


if __name__ == "__main__":
    try:
        with open("keys/canvas.txt", "r") as file:
            api_key = file.read().strip()
    except Exception as e:
        print("Error reading API key:", e)
        api_key = ""

    API_URL = "https://canvas.nus.edu.sg/"
    manager = CanvasManager(API_URL, api_key)

    # Uncomment the following line to download all files
    # manager.download_all_files_parallel(base_dir="files")

    # To build the embedding index (without generating summaries):
    manager.build_embedding_index(index_dir="chroma_index")

    # To generate summaries separately (with sensitive information removed):
    manager.build_all_summaries(base_dir="files", summary_base_dir="summary")

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
    """

    print("\nExample 5: Using filter terms 'VSD' and 'EBA5004'")
    results5 = manager.retrieve_lecture_slides_by_topic(
        topic="what is transformer?",
        filter_terms=["VSD", "EBA5004"]
    )
    print("\nFinal top results from Example 5:", results5)

    print("\nExample 6: Using no filter")
    results6 = manager.retrieve_lecture_slides_by_topic(
        topic="how to implement langchain?"
    )
    print("\nFinal top results from Example 6:", results6)

    # 2. TIMETABLE
    manager.get_timetable(True, 2024, "AIS06")

    # 3. ASSIGNMENTS & DEADLINES
    manager.list_upcoming_assignments(hide_older_than=0)
    manager.get_assignment_detail("CNI Day 4 Workshop")

    # 4. ANNOUNCEMENTS & NOTIFICATIONS
    manager.list_announcements(hide_older_than=7, only_unread=False)
    manager.get_announcement_detail("Internship Announcement")
