import os
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from canvasapi import Canvas


class CanvasManager:
    """
    A manager class to handle Canvas operations such as retrieving assignments,
    announcements, and downloading files concurrently.
    """

    def __init__(self, api_url: str, api_key: str):
        self.canvas = Canvas(api_url, api_key)

    def get_assignments_due_dates(self, hide_older_than: int = 7, max_workers: int = 5):
        """
        Retrieve assignments from all accessible courses concurrently that have a due date,
        skipping assignments without a due date or those due before (now - hide_older_than days).
        Returns a list of assignments sorted by due date.
        """
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
                    due_date_str = assignment.due_at  # ISO formatted string or None
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

    def get_announcements(self, hide_older_than: int = 7, only_unread: bool = False, max_workers: int = 5):
        """
        Retrieve announcements from all accessible courses concurrently and sort them by creation date.
        Announcements older than (now - hide_older_than days) are skipped. If only_unread is True,
        only include announcements marked as unread.
        """
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
                    # Skip if only_unread is required and the announcement is not unread.
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

    def download_all_files_parallel(self, base_dir: str = "files", max_workers: int = 5):
        """
        Download all files for accessible courses concurrently while preserving the subfolder structure.
        Files are saved to: base_dir/{course_name}/{subfolder(s)}/{filename}.
        """
        courses = list(self.canvas.get_courses())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for course in courses:
                if getattr(course, "access_restricted_by_date", False):
                    print(f"Skipping restricted course: {course.id}")
                    continue

                course_name = self.sanitize_filename(course.name)
                print(f"Processing course: {course_name}")

                folder_path_map = self.build_folder_path_map(course)
                try:
                    files = list(course.get_files())
                except Exception as e:
                    print(f"Error retrieving files for course {course.name}: {e}")
                    continue

                for f in files:
                    futures.append(
                        executor.submit(self.download_file, f, course_name, folder_path_map, base_dir)
                    )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in a download task: {e}")

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

    @staticmethod
    def remove_course_files_prefix(path: str) -> str:
        """
        Remove the top-level "course files" folder from the path if present.
        """
        if not path:
            return path
        parts = path.split(os.path.sep)
        if parts and parts[0].lower() == "course files":
            return os.path.join(*parts[1:]) if len(parts) > 1 else ""
        return path

    @classmethod
    def download_file(cls, f, course_name: str, folder_path_map: dict, base_dir: str):
        """
        Download a single file using its folder_id to determine the subfolder.
        """
        subfolder = ""
        if f.folder_id and f.folder_id in folder_path_map:
            subfolder = cls.remove_course_files_prefix(folder_path_map[f.folder_id])
        file_name = cls.sanitize_filename(f.display_name)
        local_file_path = os.path.join(base_dir, course_name, subfolder, file_name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        try:
            f.download(local_file_path)
            print(f"Downloaded: {local_file_path}")
        except Exception as e:
            print(f"Error downloading file {f.display_name}: {e}")


if __name__ == "__main__":
    # Example usage:
    with open("keys/canvas.txt", "r") as file:
        api_key = file.read().strip()

    API_URL = "https://canvas.nus.edu.sg/"
    manager = CanvasManager(API_URL, api_key)

    # Retrieve assignments concurrently.
    assignments = manager.get_assignments_due_dates(hide_older_than=7)
    print("=== Sorted Assignments by Due Date (excluding those older than 7 days and without a due date) ===")
    for a in assignments:
        print(f"{a['due_at_str']} - {a['course_name']}: {a['assignment_name']}")

    # Retrieve announcements concurrently.
    announcements = manager.get_announcements(hide_older_than=7, only_unread=False)
    print("\n=== Sorted Announcements by Date (excluding those older than 7 days) ===")
    for ann in announcements:
        print(f"{ann['created_at_str']} - {ann['course_name']}: {ann['announcement_title']}")

    # Download all files concurrently.
    manager.download_all_files_parallel(base_dir="files")
