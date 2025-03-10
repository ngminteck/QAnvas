import os
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from canvasapi import Canvas
import pprint

class CanvasManager:
    """
    A manager class to handle Canvas operations such as retrieving assignments,
    announcements, timetables, and downloading files concurrently.
    """

    def __init__(self, api_url: str, api_key: str):
        self.canvas = Canvas(api_url, api_key)

    # ========================
    # TIMETABLE FUNCTIONS
    # ========================

    def get_timetable(self, full_time: bool, intake: int, course: str):
        """
        Retrieve and display timetable files for the specified parameters.
        The expected folder structure is:
          course files\Timetable\MTech Full Time\Aug <intake> FT Intake  (for full-time)
          course files\Timetable\MTech Part Time\Jan <intake> PT Intake  (for part-time)

        Then, using the course code, only the matching timetable file(s) are shown.

        Parameters:
          full_time (bool): True for full-time, False for part-time.
          intake (int): The intake year (e.g., 2024 or 2022).
          course (str): The course code (e.g., "AIS06").
        """
        # Build folder names based on full_time flag.
        if full_time:
            mode_folder = "MTech Full Time"
            intake_folder = f"Aug {intake} FT Intake"
        else:
            mode_folder = "MTech Part Time"
            intake_folder = f"Jan {intake} PT Intake"

        # Define the timetable course name (the course that holds timetable files)
        target_course_name = "MTech in EBAC/IS/SE (Thru-train)".lower()

        try:
            courses = list(self.canvas.get_courses())
        except Exception as e:
            print("Error retrieving courses:", e)
            return

        # Locate the timetable course.
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

        # Build the target folder path as stored in Canvas.
        target_folder = f"course files\\Timetable\\{mode_folder}\\{intake_folder}"
        print(f"Looking for files in folder: {target_folder}")

        # Filter files in the target folder.
        files_in_target = []
        for f in files:
            folder_id = getattr(f, "folder_id", None)
            folder_name = folder_map.get(folder_id, None)
            if folder_name and folder_name.lower() == target_folder.lower():
                files_in_target.append(f.display_name)

        if not files_in_target:
            print(f"No files found in folder {target_folder}.")
            return

        # Further filter by course code.
        course_code = course.upper()
        files_for_course = [fname for fname in files_in_target if course_code in fname.upper()]

        if files_for_course:
            print(f"\nFiles for course code {course}:")
            for file_name in files_for_course:
                print("  -", file_name)
        else:
            print(f"No timetable file found for course code {course}.")

    # ========================
    # ASSIGNMENTS FUNCTIONS
    # ========================

    def get_assignments_due_dates(self, hide_older_than: int = 90, max_workers: int = 5):
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

    def list_upcoming_assignments(self, hide_older_than: int = 90):
        """
        List upcoming assignments based on due dates.
        """
        assignments = self.get_assignments_due_dates(hide_older_than=hide_older_than)
        if assignments:
            print("\nUpcoming Assignments:")
            for idx, a in enumerate(assignments, start=1):
                print(f"{idx}. {a['due_at_str']} - {a['course_name']}: {a['assignment_name']}")
        else:
            print("No upcoming assignments found.")
        return assignments

    def get_assignment_detail(self, assignment_name: str):
        """
        Retrieve detailed information for a specific assignment by name.
        This may include the due date and additional details available in the assignment object.
        """
        courses = list(self.canvas.get_courses())
        for course in courses:
            if getattr(course, "access_restricted_by_date", False):
                continue
            try:
                for assignment in course.get_assignments():
                    if assignment.name.lower() == assignment_name.lower():
                        print("\nAssignment Details:")
                        print(f"Course: {course.name}")
                        print(f"Assignment: {assignment.name}")
                        print(f"Due: {assignment.due_at}")
                        print("Full Details:")
                        pprint.pprint(assignment.__dict__)
                        return assignment
            except Exception as e:
                print(f"Error retrieving assignments for course {course.name}: {e}")
        print("Assignment not found.")
        return None

    # ========================
    # ANNOUNCEMENTS FUNCTIONS
    # ========================

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
                print(f"{idx}. {ann['created_at_str']} - {ann['course_name']}: {ann['announcement_title']}")
        else:
            print("No announcements found.")
        return announcements

    def get_announcement_detail(self, announcement_title: str):
        """
        Retrieve detailed content for a specific announcement by title.
        """
        courses = list(self.canvas.get_courses())
        for course in courses:
            if getattr(course, "access_restricted_by_date", False):
                continue
            try:
                for ann in course.get_discussion_topics(only_announcements=True):
                    if ann.title.lower() == announcement_title.lower():
                        print("\nAnnouncement Detail:")
                        print(f"Course: {course.name}")
                        print(f"Title: {ann.title}")
                        print(f"Created at: {getattr(ann, 'created_at', 'N/A')}")
                        detail = getattr(ann, 'message', None)
                        if not detail:
                            detail = getattr(ann, 'description', None)
                        if detail:
                            print("Content:")
                            print(detail)
                        else:
                            print("No additional content available.")
                        print("Full Details:")
                        pprint.pprint(ann.__dict__)
                        return ann
            except Exception as e:
                print(f"Error retrieving announcements for course {course.name}: {e}")
        print("Announcement not found.")
        return None

    # ========================
    # DOWNLOAD FILES FUNCTIONS
    # ========================

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

    # ========================
    # HELPER FUNCTIONS
    # ========================

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

    # 1. TIMETABLE
    print("\n=== TIMETABLE ===")
    manager.get_timetable(True, 2024, "AIS06")

    # 2. ASSIGNMENTS & DEADLINES
    print("\n=== ASSIGNMENTS & DEADLINES ===")
    manager.list_upcoming_assignments(hide_older_than=90)
    # Uncomment and replace with an actual assignment title to view details:
    # manager.get_assignment_detail("Assignment Name")

    # 3. ANNOUNCEMENTS & NOTIFICATIONS
    print("\n=== ANNOUNCEMENTS & NOTIFICATIONS ===")
    manager.list_announcements(hide_older_than=7, only_unread=False)
    # Uncomment and replace with an actual announcement title to view details:
    # manager.get_announcement_detail("Announcement Title")

    # 4. DOWNLOAD ALL FILES
    print("\n=== DOWNLOADING FILES ===")
    #manager.download_all_files_parallel(base_dir="files")
