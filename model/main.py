# sunsets/model/main.py
"""
Main entry-point for the GUI app.
"""
# Python imports
from __future__ import annotations
import sys

# 3rd-party imports
from PyQt6 import QtWidgets

# 1st-party imports
from gui import SunsetGUI


def main() -> None:
    """
    Main function for the GUI ML app.
    """
    app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
    gui: SunsetGUI = SunsetGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
