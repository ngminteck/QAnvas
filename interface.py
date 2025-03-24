import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel
)
from PyQt6.QtCore import QThread, pyqtSignal

# Import the ask_canvas_agent function from your agent module.
# Make sure that agent.py is in the same directory or in your PYTHONPATH.
from agent import ask_canvas_agent


class AgentWorker(QThread):
    """Worker thread to execute ask_canvas_agent without blocking the UI."""
    resultReady = pyqtSignal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        # Run the query against the Canvas agent and emit the result.
        result = ask_canvas_agent(self.query)
        self.resultReady.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canvas Q&A Agent Interface")
        self.resize(600, 500)
        self._setup_ui()

    def _setup_ui(self):
        # Main widget and layout.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Conversation label and display area.
        conversation_label = QLabel("Conversation:")
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)

        # Input area.
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Enter your query here...")
        self.send_button = QPushButton("Send Query")

        # Layout for input and send button.
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(self.send_button)

        # Add widgets to the main layout.
        main_layout.addWidget(conversation_label)
        main_layout.addWidget(self.conversation_display)
        main_layout.addLayout(input_layout)

        # Connect signals.
        self.send_button.clicked.connect(self.send_query)
        self.input_line.returnPressed.connect(self.send_query)

    def send_query(self):
        query = self.input_line.text().strip()
        if not query:
            return

        # Append the user's query to the conversation display.
        self.conversation_display.append(f"You: {query}")
        self.input_line.clear()
        self.send_button.setEnabled(False)

        # Start the AgentWorker thread.
        self.worker = AgentWorker(query)
        self.worker.resultReady.connect(self.display_result)
        self.worker.start()

    def display_result(self, result: str):
        # Append the agent's response to the conversation display.
        self.conversation_display.append(f"Agent: {result}")
        self.send_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
