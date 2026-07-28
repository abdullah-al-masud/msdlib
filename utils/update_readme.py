"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import os
import subprocess
import sys
from pathlib import Path
from xml.dom import minidom


def _project_root():
    return Path(__file__).resolve().parent.parent


def generate_coverage_xml(project_root):
    coverage_file = project_root / ".coverage"
    if not coverage_file.exists():
        return -1

    result = subprocess.run(
        [sys.executable, "-m", "coverage", "xml", "--ignore-errors"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    print(
        'return code for "coverage xml": %s (0 means successful)'
        % result.returncode
    )
    if result.returncode != 0 and result.stderr:
        print(result.stderr.strip())
    return result.returncode


def get_coverage(project_root):
    coverage_xml = project_root / "coverage.xml"
    dom = minidom.parse(str(coverage_xml))
    elements = dom.getElementsByTagName("coverage")

    if len(elements) > 0:
        percentage = float(elements[0].attributes["line-rate"].value) * 100
    else:
        raise ValueError("coverage.xml file doesnt contain proper coverage report")

    return percentage


def categorize_coverage_percentage(perc):
    if perc >= 90:
        color = "lightgreen"
    elif perc >= 70 and perc < 90:
        color = "green"
    elif perc >= 50 and perc < 70:
        color = "yellow"
    elif perc >= 0 and perc < 50:
        color = "red"
    else:
        raise ValueError("coverage percentage %f is not a valid percentage" % (perc))

    return color


def edit_readme_with_coverage(readme_paths, perc, color, project_root):
    for rel_path in readme_paths:
        path = project_root / rel_path
        with open(path, "r") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if "https://img.shields.io/badge/coverage" in line:
                if str(path).endswith(".md"):
                    lines[i] = (
                        "![](https://img.shields.io/badge/coverage-%d%%25-%s)\n"
                        % (perc, color)
                    )
                elif str(path).endswith(".rst"):
                    lines[i] = (
                        ".. image:: https://img.shields.io/badge/coverage-%d%%25-%s\n"
                        % (perc, color)
                    )
                break

        with open(path, "w") as file:
            file.writelines(lines)

    (project_root / "coverage.xml").unlink()


def update_code_coverage(readme_paths=None):
    project_root = _project_root()
    readme_paths = readme_paths or ["README.md", "docs/README.rst"]

    code = generate_coverage_xml(project_root)
    if code == 0:
        perc = get_coverage(project_root)
        color = categorize_coverage_percentage(perc)
        print("code coverage percentage:", round(perc), ",  badge color-", color)
        edit_readme_with_coverage(readme_paths, round(perc), color, project_root)
    elif code == -1:
        print(".coverage file is not available, skipping code coverage update")
    else:
        print("couldnt generate coverage xml file, skipping code coverage update")


if __name__ == "__main__":
    update_code_coverage()
